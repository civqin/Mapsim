from uxsim import *
from tqdm import tqdm
import csv
import pandas as pd
import shutil
import os
import pickle
import networkx as nx
from collections import defaultdict
import time
import numpy as np

def get_affected_links(closed_link_ids, links_df):
    """分析受封禁路段影响的其他路段（考虑流量和长度）"""
    # 创建有向图
    G = nx.DiGraph()
    
    # 创建路段ID到节点的映射
    link_to_nodes = {}
    
    # 添加所有路段到图中
    for _, row in links_df.iterrows():
        G.add_edge(row['start'], row['end'], 
                  link_id=row['id'],
                  capacity=row['capacity'],
                  length=row['length'])
        link_to_nodes[row['id']] = (row['start'], row['end'])
    
    affected_links = {}
    
    # 对每个封禁路段进行分析
    for closed_link_id in closed_link_ids:
        start_node, end_node = link_to_nodes[closed_link_id]
        
        # 只分析直接相邻的节点
        predecessors = set(G.predecessors(start_node))
        successors = set(G.successors(end_node))
        
        # 找到所有可能受到影响的路径
        for source in predecessors:
            for target in successors:
                try:
                    # 使用最短路径算法
                    path = nx.shortest_path(G, source, target)
                    # 检查路径是否经过封禁路段
                    for i in range(len(path)-1):
                        edge_data = G.get_edge_data(path[i], path[i+1])
                        if edge_data['link_id'] == closed_link_id:
                            # 记录路径上的所有路段
                            for j in range(len(path)-1):
                                edge_data = G.get_edge_data(path[j], path[j+1])
                                link_id = edge_data['link_id']
                                if link_id not in affected_links:
                                    # 计算影响程度
                                    # 1. 考虑路段长度的影响（距离越远影响越小）
                                    length_factor = min(1.0, 1000 / edge_data['length'])
                                    # 2. 考虑通行能力的影响（通行能力越大，影响越大）
                                    capacity_factor = min(1.0, edge_data['capacity'] / 1000)
                                    # 3. 综合计算影响程度
                                    impact_level = (length_factor + capacity_factor) / 2
                                    
                                    affected_links[link_id] = impact_level
                            break
                except nx.NetworkXNoPath:
                    continue
    
    return affected_links

def run_simulation(closed_link_ids=None, use_original_results=False):
    print("\n开始初始化模拟...")
    # 创建新的World实例
    W = World(
        name="sioufalls_matsim_closed_link",
        deltan=1,
        tmax=10500,
        print_mode=1, save_mode=1, show_mode=1,
        random_seed=0
    )

    # 复制原始的路段文件
    if closed_link_ids:
        print("处理路段封禁...")
        # 读取原始links.csv
        links_df = pd.read_csv('matsim/links.csv')
        
        # 分析受影响的路段
        affected_links = get_affected_links(closed_link_ids, links_df)
        print(f"\n受封禁路段影响的其他路段: {', '.join(affected_links.keys())}")
        
        # 一次性修改所有路段的通行能力
        mask_closed = links_df['id'].isin(closed_link_ids)
        
        # 封禁路段完全关闭
        links_df.loc[mask_closed, 'capacity'] = 0
        
        # 根据影响程度动态调整受影响路段的通行能力
        for link_id, impact_level in affected_links.items():
            if link_id not in closed_link_ids:
                # 通行能力降低到30%-60%之间，具体取决于影响程度
                new_capacity = links_df.loc[links_df['id'] == link_id, 'capacity'].iloc[0] * (0.3 + 0.3 * impact_level)
                links_df.loc[links_df['id'] == link_id, 'capacity'] = new_capacity
        
        # 保存修改后的文件
        links_df.to_csv('matsim/links_closed.csv', index=False)
        links_file = 'matsim/links_closed.csv'
    else:
        links_file = 'matsim/links.csv'

    print("生成路网...")
    W.generate_Nodes_from_csv("matsim/nodes.csv")
    W.generate_Links_from_csv(links_file)

    # 预加载公交数据
    print("添加公交需求...")
    bus_data = pd.read_csv('matsim/bus.csv')
    lines = bus_data.groupby('id')
    
    # 批量添加公交需求
    for line_id, stops in lines:
        print(f"线路: {line_id}")
        for start_time in range(60, 10561, 500):
            current_time = start_time
            for i in range(len(stops) - 1):
                current_stop = stops.iloc[i]
                next_stop = stops.iloc[i + 1]
                W.adddemand_point2point(
                    float(current_stop['x']), float(current_stop['y']),
                    float(next_stop['x']), float(next_stop['y']),
                    float(current_time), 
                    float(current_time)+W.DELTAT,
                    flow=1
                )
                current_time += current_stop['inter_stop_time']

    # 优化私家车需求添加
    print("添加私家车需求...")
    csv_file = 'matsim/population_time_converted.csv'
    
    # 使用pandas读取数据
    df = pd.read_csv(csv_file)
    
    # 修正车辆需求筛选逻辑
    car_demands = df[
        (df['mode1'] == 'car') & 
        (df['type2'] == 'work')
    ]
    
    # 按起点和终点分组，确保每个OD对只添加一次
    car_demands = car_demands.groupby(['x1', 'y1', 'x2', 'y2']).first().reset_index()
    
    print(f"\n总车辆需求数量: {len(car_demands)}")
    
    # 使用更简单的时间分配方式
    time_windows = np.linspace(0, 10500, 35)  # 将时间分成35个窗口
    vehicles_per_window = len(car_demands) // len(time_windows)
    
    # 批量添加车辆需求
    for i, row in car_demands.iterrows():
        window_idx = i // vehicles_per_window
        if window_idx >= len(time_windows):
            window_idx = len(time_windows) - 1
        start_time = time_windows[window_idx]
        
        W.adddemand_point2point(
            float(row['x1']), float(row['y1']),
            float(row['x2']), float(row['y2']),
            float(start_time), 
            float(start_time)+W.DELTAT,
            flow=1
        )

    print("\n开始运行模拟...")
    start_time = time.time()
    
    # 运行模拟
    W.exec_simulation()
    
    end_time = time.time()
    print(f"\n模拟完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 统计实际运行的车辆数量
    total_vehicles = sum(len(link.vehicles) for link in W.links)
    print(f"\n实际运行的车辆数量: {total_vehicles}")
    
    if not closed_link_ids:  # 只在原始模拟时保存结果
        print("保存原始模拟结果...")
        results = {
            'nodes': W.nodes,
            'links': W.links,
            'analyzer': W.analyzer,
            'DELTAT': W.DELTAT,
            'TMAX': W.TMAX,
            'total_vehicles': total_vehicles
        }
        with open('original_simulation.pkl', 'wb') as f:
            pickle.dump(results, f)

    return W

if __name__ == "__main__":
    try:
        # 定义要封禁的路段
        closed_links = ['11_1', '11_2', '11_3', '11_4', '11_5']
        
        # 检查是否存在原始模拟结果
        use_original = os.path.exists('original_simulation.pkl')
        
        if not use_original:
            # 运行原始模拟
            print("运行原始模拟...")
            W_original = run_simulation()
            print("\n原始模拟结果:")
            W_original.analyzer.print_simple_stats()
        else:
            print("找到原始模拟结果，跳过原始模拟...")
        
        # 运行封禁路段的模拟
        print(f"\n运行路段 {', '.join(closed_links)} 封禁的模拟...")
        W_closed = run_simulation(closed_links)
        print(f"\n路段封禁后的模拟结果:")
        W_closed.analyzer.print_simple_stats()
        
        # 可视化结果
        print("\n生成动画...")
        W_closed.analyzer.network_anim(animation_speed_inverse=15, detailed=0, network_font_size=0)
        W_closed.analyzer.network_fancy(animation_speed_inverse=15, sample_ratio=0.1, interval=10, trace_length=5, speed_coef=4)
        
        # 清理临时文件
        if os.path.exists('matsim/links_closed.csv'):
            os.remove('matsim/links_closed.csv')
            
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        print("请检查输入数据和参数是否正确") 