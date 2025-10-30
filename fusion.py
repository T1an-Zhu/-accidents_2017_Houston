import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter

def preprocess_weather_data(weather_df):
    """预处理气象预警数据"""
    # 复制数据避免修改原数据
    df = weather_df.copy()
    
    # 转换时间格式
    df['issue_datetime'] = df['issue_time'].apply(parse_weather_time)
    
    # 移除无效的时间记录
    original_count = len(df)
    df = df[df['issue_datetime'].notna()].copy()
    if len(df) < original_count:
        print(f"移除时间无效的气象预警记录: {original_count - len(df)} 条")
    
    # 提取日期信息
    df['issue_date'] = df['issue_datetime'].dt.date  # 转换为date类型
    df['issue_month'] = df['issue_datetime'].dt.month
    df['issue_day'] = df['issue_datetime'].dt.day
    df['issue_year'] = df['issue_datetime'].dt.year
    
    # 检查时间范围
    if len(df) > 0:
        print(f"气象预警时间范围: {df['issue_date'].min()} 到 {df['issue_date'].max()}")
    else:
        print("气象预警数据: 无有效时间记录")
    
    # 创建危险天气的虚拟变量
    hazard_columns = ['FOG', 'WIND', 'FREEZE', 'FLOOD', 'THUNDERSTORM', 
                     'TORNADO', 'COLD', 'COASTAL', 'RAIN']
        
    # 确保 hazard_types_str 和 advisories_str 都是字符串
    df['hazard_types_str'] = df['hazard_types'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
    df['advisories_str'] = df['advisories'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))

    for hazard in hazard_columns:
        df[f'hazard_{hazard}'] = df['hazard_types_str'].str.contains(hazard, na=False).astype(int)
    
    # 创建预警类型的虚拟变量
    advisory_columns = ['WIND_ADVISORY', 'FREEZE_WARNING', 'FLOOD_WATCH', 
                       'DENSE_FOG_ADVISORY', 'SMALL_CRAFT_ADVISORY', 'GALE_WARNING']
    
    for advisory in advisory_columns:
        df[f'advisory_{advisory}'] = df['advisories_str'].str.contains(advisory, na=False).astype(int)
    
    # 计算危险天气数量
    hazard_col_names = [f'hazard_{hazard}' for hazard in hazard_columns]
    df['hazard_count'] = df[hazard_col_names].sum(axis=1)
    
    # 标记是否有严重预警
    severe_advisory_cols = [f'advisory_{adv}' for adv in ['WIND_ADVISORY', 'FREEZE_WARNING', 'FLOOD_WATCH', 'GALE_WARNING']]
    df['has_severe_advisory'] = (df[severe_advisory_cols].sum(axis=1) > 0).astype(int)
    
    return df

import re
from datetime import datetime
import pandas as pd

def parse_weather_time(time_str):
    """解析气象预警时间，忽略时区（CST/CDT），只提取实际日期和时间"""
    if pd.isna(time_str) or time_str == 'Unknown':
        return pd.NaT

    try:
        # 匹配格式示例：
        # 429 AM CST FRI DEC 30 2017  -> 忽略 CST
        # 631 PM CDT THU JUL 20 2017  -> 忽略 CDT
        pattern = (
            r'(?P<hour_min>\d{1,4})\s+'   # 1~4位数字表示小时+分钟
            r'(?P<am_pm>[AP]M)\s+'        # AM/PM
            r'(?:[A-Z]{2,4}\s+)?'         # 可选的时区，非捕获组
            r'(?:[A-Z]{3}\s+)?'           # 可选的星期缩写
            r'(?P<month>[A-Z]{3})\s+'     # 月份缩写
            r'(?P<day>\d{1,2})\s+'        # 日
            r'(?P<year>\d{4})'            # 年
        )

        match = re.search(pattern, str(time_str).upper())
        if not match:
            return pd.to_datetime(time_str, errors='coerce')

        gd = match.groupdict()
        hour_min = gd['hour_min']
        am_pm = gd['am_pm']
        month = gd['month']
        day = int(gd['day'])
        year = int(gd['year'])

        # 分离小时和分钟
        if len(hour_min) <= 2:
            hour = int(hour_min)
            minute = 0
        else:
            hour = int(hour_min[:-2])
            minute = int(hour_min[-2:])

        # 12小时制转换为24小时制
        if am_pm == 'PM' and hour != 12:
            hour += 12
        elif am_pm == 'AM' and hour == 12:
            hour = 0

        # 月份转换
        month_dict = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        month_num = month_dict.get(month.upper(), 1)

        return datetime(year, month_num, day, hour, minute)

    except Exception as e:
        print(f"时间解析错误: {time_str}, 错误: {e}")
        return pd.NaT

def preprocess_accident_data(accident_df, allowed_months=None):
    """预处理事故数据
    
    allowed_months: 可选 - 如果提供（例如来自气象数据的 month 集合），则只保留这些月份内的事故记录，
                    以避免因为解析歧义造成的异常月份（例如意外的6或7月）。
    """
    df = accident_df.copy()
    
    # 初始解析（默认不启用 dayfirst）
    df['Start_Time_parsed'] = pd.to_datetime(df['Start_Time'], errors='coerce', dayfirst=False)
    df['End_Time_parsed'] = pd.to_datetime(df['End_Time'], errors='coerce', dayfirst=False)
    
    # 把解析结果搬到标准列并标记
    df['Start_Time'] = df['Start_Time_parsed']
    df['End_Time'] = df['End_Time_parsed']
    
    # 移除 Start_Time 无效的记录
    original_count = len(df)
    df = df[df['Start_Time'].notna()].copy()
    if len(df) < original_count:
        print(f"移除时间无效的事故记录: {original_count - len(df)} 条")
    
    # 提取日期信息
    df['accident_date'] = df['Start_Time'].dt.date  # 转换为date类型
    df['accident_hour'] = df['Start_Time'].dt.hour
    df['accident_month'] = df['Start_Time'].dt.month
    df['accident_day'] = df['Start_Time'].dt.day
    df['accident_year'] = df['Start_Time'].dt.year
    df['accident_weekday'] = df['Start_Time'].dt.weekday
    
    # 计算事故持续时间（小时）
    df['duration_hours'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 3600
    
    # 检查时间范围
    if len(df) > 0:
        print(f"事故数据时间范围: {df['accident_date'].min()} 到 {df['accident_date'].max()}")
    else:
        print("事故数据: 无有效时间记录")
    
    # 如果提供 allowed_months，则筛选出 month 在 allowed_months 的记录
    if allowed_months is not None and len(allowed_months) > 0:
        allowed_set = set(int(m) for m in allowed_months if not pd.isna(m))
        before_count = len(df)
        # 找到不在 allowed_set 的记录（可疑记录）
        suspicious = df[~df['accident_month'].isin(allowed_set)]
        if len(suspicious) > 0:
            print(f"发现 {len(suspicious)} 条事故记录的解析月份不在允许月份范围 {sorted(allowed_set)} 中。")
            # 展示部分可疑记录（便于调试）
            print("示例可疑记录 (ID, Start_Time 原文, 解析后 month):")
            # 尝试显示原始 Start_Time 值：如果原始列名是 'Start_Time'（字符串）就能拿到，否则回退
            sample_cols = ['ID', 'Start_Time', 'accident_month']
            print(suspicious[sample_cols].head(10).to_string(index=False))
            # 方案：移除这些可疑记录（避免错误的月份导致后续统计混淆）
            df = df[df['accident_month'].isin(allowed_set)].copy()
            after_count = len(df)
            print(f"已移除 {before_count - after_count} 条不在允许月份范围的事故记录。")
        else:
            print("所有事故记录的月份均在允许范围内。")
    
    return df

def merge_weather_accident_data_flexible(weather_df, accident_df, time_windows=[-1, 0]):
    """灵活的时间窗口匹配方法 - 匹配事故前一天和当天发出的预警"""
    merged_data = []
    accident_matched = set()
    
    print(f"使用灵活时间窗口匹配: {time_windows}")
    print(f"开始合并数据: 事故数={len(accident_df)}, 预警数={len(weather_df)}")
    
    for _, accident in accident_df.iterrows():
        accident_date = accident['accident_date']
        accident_id = accident['ID']
        
        for days_diff in time_windows:
            target_date = accident_date + timedelta(days=days_diff)
            
            # 查找目标日期的气象预警
            relevant_weather = weather_df[weather_df['issue_date'] == target_date]
            
            if len(relevant_weather) > 0:
                for _, weather in relevant_weather.iterrows():
                    merged_record = create_merged_record_simple(accident, weather, accident_date, abs(days_diff))
                    merged_data.append(merged_record)
                    accident_matched.add(accident_id)
    
    result_df = pd.DataFrame(merged_data)
    
    print(f"\n合并结果统计:")
    print(f"  总事故数: {len(accident_df)}")
    print(f"  匹配到预警的事故数: {len(accident_matched)}")
    print(f"  未匹配的事故数: {len(accident_df) - len(accident_matched)}")
    print(f"  匹配率: {len(accident_matched)/len(accident_df)*100:.1f}%" if len(accident_df)>0 else "  无事故数据计算匹配率")
    print(f"  总关联记录数: {len(result_df)}")
    
    if len(accident_matched) > 0:
        print(f"  平均每个事故关联的气象预警数: {len(result_df)/len(accident_matched):.2f}")
    
    return result_df

def create_merged_record_simple(accident, weather, accident_date, time_diff):
    """创建合并记录（简化版）"""
    return {
        'accident_id': accident['ID'],
        'accident_severity': accident['Severity'],
        'accident_date': accident_date,
        'accident_city': accident.get('City', 'Unknown'),
        'accident_month': accident['accident_month'],
        'accident_hour': accident['accident_hour'],
        'accident_weekday': accident['accident_weekday'],
        'temperature': accident.get('Temperature(F)', None),
        'humidity': accident.get('Humidity(%)', None),
        'visibility': accident.get('Visibility(mi)', None),
        'wind_speed': accident.get('Wind_Speed(mph)', None),
        'precipitation': accident.get('Precipitation(in)', None),
        'weather_condition': accident.get('Weather_Condition', 'Unknown'),
        'distance': accident.get('Distance(mi)', None),
        'duration_hours': accident.get('duration_hours', None),
        
        'weather_issue_time': weather['issue_time'],
        'weather_issue_date': weather['issue_date'],
        'days_before_accident': time_diff,
        'coastal_waters': weather.get('coastal_waters', None),
        'hazard_count': weather.get('hazard_count', None),
        'has_severe_advisory': weather.get('has_severe_advisory', None),
        
        # 危险天气类型
        'hazard_FOG': weather.get('hazard_FOG', 0),
        'hazard_WIND': weather.get('hazard_WIND', 0),
        'hazard_FREEZE': weather.get('hazard_FREEZE', 0),
        'hazard_FLOOD': weather.get('hazard_FLOOD', 0),
        'hazard_THUNDERSTORM': weather.get('hazard_THUNDERSTORM', 0),
        'hazard_TORNADO': weather.get('hazard_TORNADO', 0),
        'hazard_COLD': weather.get('hazard_COLD', 0),
        'hazard_COASTAL': weather.get('hazard_COASTAL', 0),
        'hazard_RAIN': weather.get('hazard_RAIN', 0),
        
        # 预警类型
        'advisory_WIND_ADVISORY': weather.get('advisory_WIND_ADVISORY', 0),
        'advisory_FREEZE_WARNING': weather.get('advisory_FREEZE_WARNING', 0),
        'advisory_FLOOD_WATCH': weather.get('advisory_FLOOD_WATCH', 0),
        'advisory_DENSE_FOG_ADVISORY': weather.get('advisory_DENSE_FOG_ADVISORY', 0),
        'advisory_SMALL_CRAFT_ADVISORY': weather.get('advisory_SMALL_CRAFT_ADVISORY', 0),
        'advisory_GALE_WARNING': weather.get('advisory_GALE_WARNING', 0),
        
        # 匹配信息
        'match_type': 'flexible_time_window',
        
        # 原始文本信息
        'day_one_text': str(weather.get('day_one', ''))[:200] if pd.notna(weather.get('day_one', None)) else ''
    }

def analyze_matching_quality(weather_df, accident_df):
    """分析匹配质量"""
    print("\n" + "="*60)
    print("数据匹配质量分析")
    print("="*60)
    
    # 时间范围分析
    weather_dates = weather_df['issue_date'].dropna()
    accident_dates = accident_df['accident_date'].dropna()
    
    if len(weather_dates) > 0 and len(accident_dates) > 0:
        print(f"气象预警时间范围: {weather_dates.min()} 到 {weather_dates.max()}")
        print(f"事故时间范围: {accident_dates.min()} 到 {accident_dates.max()}")
        print(f"气象预警天数: {weather_dates.nunique()}")
        print(f"事故天数: {accident_dates.nunique()}")
    
    # 月度分布
    print(f"\n月度分布:")
    weather_monthly = weather_df['issue_month'].value_counts().sort_index()
    accident_monthly = accident_df['accident_month'].value_counts().sort_index()
    
    for month in range(1, 13):
        w_count = weather_monthly.get(month, 0)
        a_count = accident_monthly.get(month, 0)
        if w_count > 0 or a_count > 0:
            print(f"  月份 {month}: 预警={w_count}, 事故={a_count}")
    
def analyze_weather_accident_relationship(merged_df):
    """分析气象预警与事故的关系"""
    analysis_results = {}
    
    print("\n" + "="*60)
    print("气象预警与交通事故关系分析")
    print("="*60)
    
    if len(merged_df) == 0:
        print("没有数据可分析")
        return analysis_results
    
    # 基本统计
    total_accidents = len(merged_df['accident_id'].unique())
    total_merged_records = len(merged_df)
    
    print(f"\n1. 基本统计:")
    print(f"   总事故数: {total_accidents}")
    print(f"   总关联记录数: {total_merged_records}")
    if total_accidents > 0:
        print(f"   平均每个事故关联的气象预警数: {total_merged_records/total_accidents:.2f}")
    
    # 按事故严重程度分析
    print(f"\n2. 事故严重程度分布:")
    severity_counts = merged_df['accident_severity'].value_counts().sort_index()
    for severity, count in severity_counts.items():
        percentage = count / total_merged_records * 100 if total_merged_records > 0 else 0
        print(f"   严重程度 {severity}: {count} 次 ({percentage:.1f}%)")
    
    # 危险天气类型与事故关系
    print(f"\n3. 危险天气类型与事故关联:")
    hazard_columns = [col for col in merged_df.columns if col.startswith('hazard_')]
    
    for hazard in hazard_columns:
        hazard_name = hazard.replace('hazard_', '')
        hazard_count = merged_df[merged_df[hazard] == 1].shape[0]
        if hazard_count > 0:
            hazard_rate = hazard_count / total_merged_records * 100
            avg_severity = merged_df[merged_df[hazard] == 1]['accident_severity'].mean()
            print(f"   {hazard_name}: {hazard_count}次关联 ({hazard_rate:.1f}%), 平均严重程度: {avg_severity:.2f}")
    
    # 预警类型与事故关系
    print(f"\n4. 预警类型与事故关联:")
    advisory_columns = [col for col in merged_df.columns if col.startswith('advisory_')]
    
    for advisory in advisory_columns:
        advisory_name = advisory.replace('advisory_', '')
        advisory_count = merged_df[merged_df[advisory] == 1].shape[0]
        if advisory_count > 0:
            advisory_rate = advisory_count / total_merged_records * 100
            avg_severity = merged_df[merged_df[advisory] == 1]['accident_severity'].mean()
            print(f"   {advisory_name}: {advisory_count}次关联 ({advisory_rate:.1f}%), 平均严重程度: {avg_severity:.2f}")
    
    # 按时间差分析
    print(f"\n5. 预警发布与事故时间关系:")
    if 'days_before_accident' in merged_df.columns:
        time_diff_stats = merged_df.groupby('days_before_accident').agg({
            'accident_severity': 'mean',
            'accident_id': 'count'
        }).rename(columns={'accident_id': 'accident_count'}).sort_index()
        
        for days_diff, row in time_diff_stats.iterrows():
            time_desc = "事故当天" if days_diff == 0 else f"事故前{days_diff}天"
            print(f"   {time_desc}: {row['accident_count']} 起事故, 平均严重程度: {row['accident_severity']:.2f}")
    
    # 气象条件与事故严重程度相关性
    print(f"\n6. 气象条件与事故严重程度相关性:")
    numeric_columns = ['temperature', 'humidity', 'visibility', 'wind_speed', 'precipitation', 'hazard_count']
    
    for col in numeric_columns:
        if col in merged_df.columns:
            try:
                valid_data = merged_df[[col, 'accident_severity']].dropna()
                if len(valid_data) > 1:
                    correlation = valid_data[col].corr(valid_data['accident_severity'])
                    print(f"   {col}: 相关性系数 = {correlation:.3f}")
            except:
                print(f"   {col}: 无法计算相关性")
    
    return analysis_results

def create_analysis_report(merged_df, output_file="./data/weather_accident_analysis_report.txt"):
    """创建详细分析报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("气象预警与交通事故关联分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 基本统计
        f.write("1. 数据概览\n")
        f.write(f"   总关联记录数: {len(merged_df)}\n")
        if len(merged_df) > 0:
            f.write(f"   唯一事故数: {merged_df['accident_id'].nunique()}\n")
            f.write(f"   覆盖时间范围: {merged_df['accident_date'].min()} 到 {merged_df['accident_date'].max()}\n\n")
        else:
            f.write("   无有效数据\n\n")
        
        # 事故严重程度分析
        f.write("2. 事故严重程度分析\n")
        if len(merged_df) > 0:
            severity_summary = merged_df.groupby('accident_severity').agg({
                'accident_id': 'count',
                'duration_hours': 'mean',
                'hazard_count': 'mean'
            }).round(3)
            
            for severity, row in severity_summary.iterrows():
                f.write(f"   严重程度 {severity}: {row['accident_id']} 起事故, ")
                f.write(f"平均持续时间: {row['duration_hours']:.2f} 小时, ")
                f.write(f"平均危险天气数: {row['hazard_count']:.2f}\n")
        f.write("\n")
        
        # 危险天气影响分析
        f.write("3. 危险天气类型影响分析\n")
        if len(merged_df) > 0:
            hazard_impact = {}
            hazard_columns = [col for col in merged_df.columns if col.startswith('hazard_')]
            
            for hazard in hazard_columns:
                hazard_data = merged_df[merged_df[hazard] == 1]
                if len(hazard_data) > 0:
                    avg_severity = hazard_data['accident_severity'].mean()
                    count = len(hazard_data)
                    hazard_impact[hazard] = (avg_severity, count)
            
            for hazard, (severity, count) in sorted(hazard_impact.items(), key=lambda x: x[1][0], reverse=True):
                f.write(f"   {hazard}: 平均事故严重程度 {severity:.3f} ({count}次)\n")
        f.write("\n")
        
        # 时间模式分析
        f.write("4. 时间模式分析\n")
        if len(merged_df) > 0 and 'accident_month' in merged_df.columns:
            monthly_accidents = merged_df.groupby('accident_month')['accident_id'].nunique()
            f.write("   月度事故分布:\n")
            for month, count in monthly_accidents.items():
                f.write(f"   月份 {month}: {count} 起事故\n")
        
        if len(merged_df) > 0 and 'accident_hour' in merged_df.columns:
            f.write("\n   小时事故分布 (高峰时段):\n")
            hourly_accidents = merged_df.groupby('accident_hour')['accident_id'].nunique()
            for hour, count in hourly_accidents.nlargest(6).items():
                f.write(f"   {hour:02d}:00时: {count} 起事故\n")
        
        if len(merged_df) > 0 and 'accident_weekday' in merged_df.columns:
            f.write("\n   星期分布:\n")
            weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            weekday_accidents = merged_df.groupby('accident_weekday')['accident_id'].nunique()
            for weekday, count in weekday_accidents.items():
                if weekday < len(weekday_names):
                    f.write(f"   {weekday_names[weekday]}: {count} 起事故\n")
    
    print(f"详细分析报告已保存: {output_file}")

def debug_data_structure(weather_df, accident_df):
    """调试数据结构"""
    print("气象数据列名:", weather_df.columns.tolist())
    print("气象数据形状:", weather_df.shape)
    print("\n事故数据列名:", accident_df.columns.tolist())
    print("事故数据形状:", accident_df.shape)
    
    # 检查关键列是否存在
    required_weather_cols = ['hazard_types_str', 'advisories_str', 'issue_time']
    for col in required_weather_cols:
        if col in weather_df.columns:
            print(f"✓ 气象数据包含列: {col}")
        else:
            print(f"✗ 气象数据缺少列: {col}")

# 主程序
def main():
    # 读取数据
    print("读取数据...")
    
    try:
        # 读取数据
        weather_df = pd.read_csv("./data/weather_outlook_analysis.csv")
        accident_df = pd.read_csv("./data/USAccidents_2017_Houston_cleaned.csv")
        
        # 调试数据结构
        debug_data_structure(weather_df, accident_df)
        
        # 数据预处理 - 先处理气象数据以获得允许的月份（用于事故数据过滤）
        print("\n预处理气象数据...")
        weather_processed = preprocess_weather_data(weather_df)
        allowed_months = sorted(weather_processed['issue_month'].dropna().unique().tolist())
        print(f"气象数据中出现的月份: {allowed_months}")
        
        # 预处理事故数据，传入 allowed_months 以避开解析歧义导致的异常月份
        print("\n预处理事故数据...")
        accident_processed = preprocess_accident_data(accident_df, allowed_months=allowed_months)
        
        print(f"\n原始数据统计:")
        print(f"  气象预警记录数: {len(weather_processed)}")
        print(f"  事故记录数: {len(accident_processed)}")
        
        # 分析匹配质量
        analyze_matching_quality(weather_processed, accident_processed)
        
        # 使用灵活时间窗口匹配方法（事故前一天和当天）
        print("\n使用灵活时间窗口匹配方法（事故前一天和当天）...")
        merged_df = merge_weather_accident_data_flexible(
            weather_processed, accident_processed, 
            time_windows=[-1, 0]  # 事故前一天和当天
        )
        
        if len(merged_df) > 0:
            # 分析关系
            print("\n进行分析...")
            analysis_results = analyze_weather_accident_relationship(merged_df)
            
            # 创建详细报告
            create_analysis_report(merged_df)
            
            # 保存合并后的数据
            merged_df.to_csv("./data/weather_accident_merged_houston.csv", index=False, encoding='utf-8')
            print(f"\n合并数据已保存: weather_accident_merged_houston.csv")
            
            from fusion_visualisation import visualize_selected_weather_accident_data
            visualize_selected_weather_accident_data(merged_df)
        else:
            print("没有找到匹配的记录")
            
    except FileNotFoundError as e:
        print(f"文件读取错误: {e}")
        print("请检查文件路径是否正确")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
