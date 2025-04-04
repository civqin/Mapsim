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
    """分析受封禁路段影响的其他路段（基于路径推理）"""
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
        
        # 第一层：直接相连的路段
        first_layer_links = set()
        
        # 找到所有进入起点节点的路段
        for pred in G.predecessors(start_node):
            edge_data = G.get_edge_data(pred, start_node)
            first_layer_links.add(edge_data['link_id'])
        
        # 找到所有从终点节点出发的路段
        for succ in G.successors(end_node):
            edge_data = G.get_edge_data(end_node, succ)
            first_layer_links.add(edge_data['link_id'])
        
        # 第二层：与第一层路段相连的路段
        second_layer_links = set()
        
        # 遍历第一层路段，找到它们相连的路段
        for first_layer_link in first_layer_links:
            first_start, first_end = link_to_nodes[first_layer_link]
            
            # 找到进入第一层路段起点的路段
            for pred in G.predecessors(first_start):
                edge_data = G.get_edge_data(pred, first_start)
                if edge_data['link_id'] not in first_layer_links:
                    second_layer_links.add(edge_data['link_id'])
            
            # 找到从第一层路段终点出发的路段
            for succ in G.successors(first_end):
                edge_data = G.get_edge_data(first_end, succ)
                if edge_data['link_id'] not in first_layer_links:
                    second_layer_links.add(edge_data['link_id'])
        
        # 分析可能的绕行路径
        # 1. 找到所有可能使用封禁路段的OD对
        potential_od_pairs = []
        
        # 获取所有可能使用封禁路段的起点
        for pred in G.predecessors(start_node):
            potential_od_pairs.append((pred, end_node))
        
        # 获取所有可能使用封禁路段的终点
        for succ in G.successors(end_node):
            potential_od_pairs.append((start_node, succ))
        
        # 2. 计算绕行路径
        link_usage_count = defaultdict(int)
        
        # 临时移除封禁路段
        G.remove_edge(start_node, end_node)
        
        for origin, destination in potential_od_pairs:
            try:
                # 找到最短绕行路径
                path = nx.shortest_path(G, origin, destination, weight='length')
                
                # 统计路径上每个路段的使用次数
                for i in range(len(path)-1):
                    edge_data = G.get_edge_data(path[i], path[i+1])
                    link_usage_count[edge_data['link_id']] += 1
            except nx.NetworkXNoPath:
                continue
        
        # 恢复封禁路段
        G.add_edge(start_node, end_node, 
                  link_id=closed_link_id,
                  capacity=0,
                  length=G.get_edge_data(start_node, end_node)['length'])
        
        # 计算总使用次数
        total_usage = sum(link_usage_count.values())
        
        # 为第一层路段分配影响程度（基于路径使用频率）
        for link_id in first_layer_links:
            if link_id not in closed_link_ids:
                # 计算该路段在绕行路径中的使用频率
                usage_ratio = link_usage_count[link_id] / total_usage if total_usage > 0 else 0
                # 根据使用频率计算影响程度（0.5-0.9）
                impact_level = 0.5 + 0.4 * usage_ratio
                affected_links[link_id] = impact_level
        
        # 为第二层路段分配影响程度（基于路径使用频率）
        for link_id in second_layer_links:
            if link_id not in closed_link_ids:
                # 计算该路段在绕行路径中的使用频率
                usage_ratio = link_usage_count[link_id] / total_usage if total_usage > 0 else 0
                # 根据使用频率计算影响程度（0.3-0.5）
                impact_level = 0.3 + 0.2 * usage_ratio
                affected_links[link_id] = impact_level
    
    return affected_links

def run_simulation(closed_link_ids=None, use_original_results=False):
    try:
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
            try:
                # 读取原始links.csv
                links_df = pd.read_csv('matsim/links.csv')
                print(f"成功读取links.csv，共{len(links_df)}条路段")
                
                # 分析受影响的路段
                affected_links = get_affected_links(closed_link_ids, links_df)
                print(f"\n受封禁路段影响的其他路段: {', '.join(affected_links.keys())}")
                
                # 一次性修改所有路段的通行能力
                mask_closed = links_df['id'].isin(closed_link_ids)
                print(f"封禁路段数量: {sum(mask_closed)}")
                
                # 封禁路段完全关闭
                links_df.loc[mask_closed, 'capacity'] = 0
                
                # 根据影响程度动态调整受影响路段的通行能力
                for link_id, impact_level in affected_links.items():
                    if link_id not in closed_link_ids:
                        # 通行能力降低到30%-60%之间，具体取决于影响程度
                        original_capacity = links_df.loc[links_df['id'] == link_id, 'capacity'].iloc[0]
                        new_capacity = original_capacity * (0.3 + 0.3 * impact_level)
                        links_df.loc[links_df['id'] == link_id, 'capacity'] = new_capacity
                        print(f"路段 {link_id}: 通行能力从 {original_capacity} 调整到 {new_capacity:.2f}")
                
                # 保存修改后的文件
                links_df.to_csv('matsim/links_closed.csv', index=False)
                links_file = 'matsim/links_closed.csv'
                print("成功保存修改后的路段文件")
            except Exception as e:
                print(f"处理路段封禁时出错: {str(e)}")
                raise
        else:
            links_file = 'matsim/links.csv'

        print("生成路网...")
        try:
            W.generate_Nodes_from_csv("matsim/nodes.csv")
            print("成功生成节点")
            W.generate_Links_from_csv(links_file)
            print("成功生成路段")
        except Exception as e:
            print(f"生成路网时出错: {str(e)}")
            raise

        # 预加载公交数据
        print("添加公交需求...")
        try:
            bus_data = pd.read_csv('matsim/bus.csv')
            lines = bus_data.groupby('id')
            print(f"成功读取公交数据，共{len(lines)}条线路")
            
            # 批量添加公交需求
            for line_id, stops in lines:
                print(f"处理线路: {line_id}")
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
        except Exception as e:
            print(f"添加公交需求时出错: {str(e)}")
            raise

        # 优化私家车需求添加
        print("添加私家车需求...")
        try:
            csv_file = 'matsim/population_time_converted.csv'
            df = pd.read_csv(csv_file)
            print(f"成功读取私家车需求数据，共{len(df)}条记录")
            
            # 修正车辆需求筛选逻辑
            car_demands = df[
                (df['mode1'] == 'car') & 
                (df['type2'] == 'work')
            ]
            print(f"筛选后的私家车需求: {len(car_demands)}条")
            
            # 按起点和终点分组，确保每个OD对只添加一次
            car_demands = car_demands.groupby(['x1', 'y1', 'x2', 'y2']).first().reset_index()
            print(f"去重后的私家车需求: {len(car_demands)}条")
            
            # 使用更简单的时间分配方式
            time_windows = np.linspace(0, 10500, 35)  # 将时间分成35个窗口
            vehicles_per_window = len(car_demands) // len(time_windows)
            print(f"每个时间窗口平均车辆数: {vehicles_per_window}")
            
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
        except Exception as e:
            print(f"添加私家车需求时出错: {str(e)}")
            raise

        print("\n开始运行模拟...")
        start_time = time.time()
        
        try:
            # 运行模拟
            W.exec_simulation()
        except Exception as e:
            print(f"模拟运行时出错: {str(e)}")
            raise
        
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
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        print("错误详情:")
        import traceback
        traceback.print_exc()
        print("\n请检查输入数据和参数是否正确")
        raise

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