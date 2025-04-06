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

def calculate_newell_propagation(closed_link_ids, links_df, demand_df):
    """
    使用Newell排队模型计算延迟传播
    
    参数:
    closed_link_ids: 被封禁的路段ID列表
    links_df: 路段数据
    demand_df: 需求数据
    """
    print("\n开始计算Newell排队模型的延迟传播...")
    
    # 创建有向图
    G = nx.DiGraph()
    link_to_nodes = {}
    
    # 添加所有路段到图中
    for _, row in links_df.iterrows():
        G.add_edge(row['start'], row['end'], 
                  link_id=row['id'],
                  capacity=row['capacity'],
                  length=row['length'],
                  free_flow_speed=row['u'],
                  jam_density=row['kappa'])
        link_to_nodes[row['id']] = (row['start'], row['end'])
    
    # 初始化延迟传播结果
    delay_propagation = defaultdict(float)
    
    # 对每个封禁路段进行分析
    for closed_link_id in closed_link_ids:
        start_node, end_node = link_to_nodes[closed_link_id]
        
        # 获取封禁路段的上游路段
        upstream_links = []
        for pred in G.predecessors(start_node):
            edge_data = G.get_edge_data(pred, start_node)
            upstream_links.append(edge_data['link_id'])
        
        # 计算封禁路段的初始延迟
        closed_link = links_df[links_df['id'] == closed_link_id].iloc[0]
        initial_delay = closed_link['length'] / closed_link['u']  # 自由流时间
        
        # 使用Newell模型计算延迟传播
        for upstream_link_id in upstream_links:
            upstream_link = links_df[links_df['id'] == upstream_link_id].iloc[0]
            
            # 计算传播参数
            free_flow_speed = upstream_link['u']
            jam_density = upstream_link['kappa']
            length = upstream_link['length']
            
            # 计算波速（根据Newell模型）
            wave_speed = free_flow_speed * jam_density / (1 - jam_density)
            
            # 计算传播时间
            propagation_time = length / abs(wave_speed)
            
            # 计算传播后的延迟
            propagated_delay = initial_delay * np.exp(-propagation_time)
            
            # 更新上游路段的延迟
            delay_propagation[upstream_link_id] = max(
                delay_propagation[upstream_link_id],
                propagated_delay
            )
            
            print(f"路段 {upstream_link_id} 的传播延迟: {propagated_delay:.2f} 秒")
    
    return delay_propagation

def adjust_link_flow_based_on_od(closed_link_ids, links_df, demand_df):
    """
    根据OD对数量和延迟传播修改路段流量
    
    参数:
    closed_link_ids: 被封禁的路段ID列表
    links_df: 路段数据
    demand_df: 需求数据
    """
    print("\n开始根据OD对数量和延迟传播调整路段流量...")
    
    # 计算延迟传播
    delay_propagation = calculate_newell_propagation(closed_link_ids, links_df, demand_df)
    
    # 创建有向图
    G = nx.DiGraph()
    link_to_nodes = {}
    
    # 添加所有路段到图中
    for _, row in links_df.iterrows():
        G.add_edge(row['start'], row['end'], 
                  link_id=row['id'],
                  capacity=row['capacity'],
                  length=row['length'])
        link_to_nodes[row['id']] = (row['start'], row['end'])
    
    # 统计每个路段的OD对数量
    link_od_count = defaultdict(int)
    link_flow = defaultdict(float)
    
    # 对每个封禁路段进行分析
    for closed_link_id in closed_link_ids:
        start_node, end_node = link_to_nodes[closed_link_id]
        
        # 找到所有可能使用封禁路段的OD对
        potential_od_pairs = []
        
        # 获取所有可能使用封禁路段的起点
        for pred in G.predecessors(start_node):
            potential_od_pairs.append((pred, end_node))
        
        # 获取所有可能使用封禁路段的终点
        for succ in G.successors(end_node):
            potential_od_pairs.append((start_node, succ))
        
        # 临时移除封禁路段
        G.remove_edge(start_node, end_node)
        
        # 对每个OD对计算绕行路径
        for origin, destination in potential_od_pairs:
            try:
                # 找到最短绕行路径
                path = nx.shortest_path(G, origin, destination, weight='length')
                
                # 获取该OD对的需求流量
                od_flow = demand_df[
                    (demand_df['orig'] == origin) & 
                    (demand_df['dest'] == destination)
                ]['q'].sum()
                
                # 统计路径上每个路段的使用次数和流量
                for i in range(len(path)-1):
                    edge_data = G.get_edge_data(path[i], path[i+1])
                    link_id = edge_data['link_id']
                    link_od_count[link_id] += 1
                    link_flow[link_id] += od_flow
            except nx.NetworkXNoPath:
                continue
        
        # 恢复封禁路段
        G.add_edge(start_node, end_node, 
                  link_id=closed_link_id,
                  capacity=0,
                  length=G.get_edge_data(start_node, end_node)['length'])
    
    # 计算总OD对数量和总流量
    total_od_count = sum(link_od_count.values())
    total_flow = sum(link_flow.values())
    
    # 修改路段的流量
    for link_id in link_od_count:
        if link_id not in closed_link_ids:
            # 计算该路段在绕行路径中的使用比例
            od_ratio = link_od_count[link_id] / total_od_count if total_od_count > 0 else 0
            flow_ratio = link_flow[link_id] / total_flow if total_flow > 0 else 0
            
            # 考虑延迟传播的影响
            delay_factor = 1.0
            if link_id in delay_propagation:
                # 延迟越大，流量调整越大
                delay_factor = 1.0 + delay_propagation[link_id] / 3600  # 转换为小时
            
            # 根据使用比例和延迟传播计算新的流量
            original_flow = links_df.loc[links_df['id'] == link_id, 'q'].iloc[0]
            new_flow = original_flow * (1 + 0.5 * (od_ratio + flow_ratio)) * delay_factor
            
            # 更新路段流量
            links_df.loc[links_df['id'] == link_id, 'q'] = new_flow
            print(f"路段 {link_id}: 流量从 {original_flow:.4f} 调整到 {new_flow:.4f}")
    
    return links_df

def run_simulation(closed_link_ids=None, use_original_results=False):
    try:
        print("\n开始初始化模拟...")
        # 创建新的World实例
        W = World(
            name="sioufalls_matsim_closed_link_with_newell",
            deltan=1,
            tmax=10500,
            print_mode=1, save_mode=1, show_mode=1,
            random_seed=0
        )

        # 复制原始的路段文件
        if closed_link_ids:
            print("处理路段封禁...")
            try:
                # 读取原始links.csv和demand.csv
                links_df = pd.read_csv('matsim/links.csv')
                demand_df = pd.read_csv('matsim/demand.csv')
                print(f"成功读取links.csv，共{len(links_df)}条路段")
                print(f"成功读取demand.csv，共{len(demand_df)}条需求")
                
                # 分析受影响的路段并调整流量
                links_df = adjust_link_flow_based_on_od(closed_link_ids, links_df, demand_df)
                
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

        print("开始模拟...")
        try:
            W.exec_simulation()
            print("模拟完成")
            
            # 保存模拟结果
            results = {
                'links': W.LINKS,
                'nodes': W.NODES,
                'vehicles': W.VEHICLES,
                'TMAX': W.TMAX,
                'DELTAT': W.DELTAT
            }
            
            with open('original_simulation.pkl', 'wb') as f:
                pickle.dump(results, f)
            print("模拟结果已保存")
            
            return results
        except Exception as e:
            print(f"模拟过程中出错: {str(e)}")
            raise
    except Exception as e:
        print(f"模拟初始化时出错: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置封禁的路段ID
    closed_link_ids = ['11']  # 示例：封禁11号路段
    
    # 运行模拟
    results = run_simulation(closed_link_ids) 