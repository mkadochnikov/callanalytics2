#!/usr/bin/env python3
"""
Функции для анализа данных по менеджерам
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def analyze_calls_by_manager(calls_data):
    """
    Анализирует звонки по менеджерам
    
    Args:
        calls_data: список словарей с данными о звонках
        
    Returns:
        DataFrame с агрегированными данными по менеджерам
    """
    manager_stats = {}
    
    for call in calls_data:
        manager_name = call.get('user_name', 'Неизвестный менеджер')
        call_direction = call.get('call_direction', 'unknown')
        
        if manager_name not in manager_stats:
            manager_stats[manager_name] = {
                'incoming': 0,
                'outgoing': 0,
                'unknown': 0,
                'total': 0
            }
        
        # Подсчитываем звонки по типам
        if call_direction == 'incoming':
            manager_stats[manager_name]['incoming'] += 1
        elif call_direction == 'outgoing':
            manager_stats[manager_name]['outgoing'] += 1
        else:
            manager_stats[manager_name]['unknown'] += 1
            
        manager_stats[manager_name]['total'] += 1
    
    # Преобразуем в DataFrame
    df_data = []
    for manager, stats in manager_stats.items():
        df_data.append({
            'Менеджер': manager,
            'Входящие': stats['incoming'],
            'Исходящие': stats['outgoing'],
            'Неопределенные': stats['unknown'],
            'Всего': stats['total']
        })
    
    df = pd.DataFrame(df_data)
    # Сортируем по общему количеству звонков
    df = df.sort_values('Всего', ascending=False)
    
    return df


def create_manager_calls_chart(df_managers):
    """
    Создает график звонков по менеджерам
    
    Args:
        df_managers: DataFrame с данными по менеджерам
        
    Returns:
        Plotly figure объект
    """
    # Создаем grouped bar chart
    fig = go.Figure()
    
    # Добавляем столбцы для входящих звонков
    fig.add_trace(go.Bar(
        name='Входящие',
        x=df_managers['Менеджер'],
        y=df_managers['Входящие'],
        marker_color='#2E8B57',  # Зеленый цвет
        text=df_managers['Входящие'],
        textposition='auto',
    ))
    
    # Добавляем столбцы для исходящих звонков
    fig.add_trace(go.Bar(
        name='Исходящие',
        x=df_managers['Менеджер'],
        y=df_managers['Исходящие'],
        marker_color='#4169E1',  # Синий цвет
        text=df_managers['Исходящие'],
        textposition='auto',
    ))
    
    # Добавляем столбцы для неопределенных звонков (если есть)
    if df_managers['Неопределенные'].sum() > 0:
        fig.add_trace(go.Bar(
            name='Неопределенные',
            x=df_managers['Менеджер'],
            y=df_managers['Неопределенные'],
            marker_color='#FFA500',  # Оранжевый цвет
            text=df_managers['Неопределенные'],
            textposition='auto',
        ))
    
    # Настраиваем макет
    fig.update_layout(
        title={
            'text': 'Количество звонков по менеджерам',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Менеджеры',
        yaxis_title='Количество звонков',
        barmode='group',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=80, b=50, l=50, r=50)
    )
    
    # Поворачиваем подписи на оси X если менеджеров много
    if len(df_managers) > 5:
        fig.update_xaxes(tickangle=45)
    
    return fig


def create_manager_total_calls_chart(df_managers):
    """
    Создает круговую диаграмму общего количества звонков по менеджерам
    
    Args:
        df_managers: DataFrame с данными по менеджерам
        
    Returns:
        Plotly figure объект
    """
    fig = px.pie(
        df_managers, 
        values='Всего', 
        names='Менеджер',
        title='Распределение общего количества звонков по менеджерам'
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Звонков: %{value}<br>Процент: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        height=400,
        title={
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        }
    )
    
    return fig


def create_manager_performance_table(df_managers):
    """
    Создает таблицу производительности менеджеров
    
    Args:
        df_managers: DataFrame с данными по менеджерам
        
    Returns:
        DataFrame для отображения в Streamlit
    """
    # Добавляем процентные показатели
    df_performance = df_managers.copy()
    
    # Процент входящих от общего количества
    df_performance['% Входящих'] = (df_performance['Входящие'] / df_performance['Всего'] * 100).round(1)
    
    # Процент исходящих от общего количества  
    df_performance['% Исходящих'] = (df_performance['Исходящие'] / df_performance['Всего'] * 100).round(1)
    
    # Соотношение исходящих к входящим
    df_performance['Соотношение И/В'] = (df_performance['Исходящие'] / df_performance['Входящие'].replace(0, 1)).round(2)
    
    return df_performance


def show_manager_analytics(calls_data):
    """
    Отображает аналитику по менеджерам в Streamlit
    
    Args:
        calls_data: список словарей с данными о звонках
    """
    import streamlit as st
    
    st.header("👥 Аналитика по менеджерам")
    
    # Получаем данные по менеджерам
    df_managers = analyze_calls_by_manager(calls_data)
    
    if df_managers.empty:
        st.warning("Нет данных для анализа по менеджерам")
        return
    
    # Показываем основные метрики
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Всего менеджеров", len(df_managers))
    
    with col2:
        avg_calls = df_managers['Всего'].mean()
        st.metric("Среднее звонков на менеджера", f"{avg_calls:.1f}")
    
    with col3:
        top_manager = df_managers.iloc[0]['Менеджер'] if len(df_managers) > 0 else "Нет данных"
        top_calls = df_managers.iloc[0]['Всего'] if len(df_managers) > 0 else 0
        st.metric("Топ менеджер", f"{top_manager} ({top_calls})")
    
    with col4:
        total_incoming = df_managers['Входящие'].sum()
        total_outgoing = df_managers['Исходящие'].sum()
        ratio = total_outgoing / total_incoming if total_incoming > 0 else 0
        st.metric("Общее соотношение И/В", f"{ratio:.2f}")
    
    # Создаем вкладки для разных представлений
    tab1, tab2, tab3 = st.tabs(["📊 График по типам", "🥧 Общее распределение", "📋 Таблица производительности"])
    
    with tab1:
        # График с разбивкой по типам звонков
        fig_bar = create_manager_calls_chart(df_managers)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        # Круговая диаграмма общего количества
        fig_pie = create_manager_total_calls_chart(df_managers)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab3:
        # Таблица производительности
        df_performance = create_manager_performance_table(df_managers)
        st.dataframe(df_performance, use_container_width=True, hide_index=True)
        
        # Дополнительные инсайты
        st.subheader("📈 Инсайты")
        
        # Самый активный менеджер
        most_active = df_performance.iloc[0]
        st.info(f"🏆 **Самый активный менеджер:** {most_active['Менеджер']} с {most_active['Всего']} звонками")
        
        # Менеджер с лучшим соотношением исходящих/входящих
        best_ratio_idx = df_performance['Соотношение И/В'].idxmax()
        best_ratio_manager = df_performance.iloc[best_ratio_idx]
        st.info(f"📞 **Лучшее соотношение И/В:** {best_ratio_manager['Менеджер']} ({best_ratio_manager['Соотношение И/В']})")
        
        # Менеджер с наибольшим процентом исходящих
        best_outgoing_idx = df_performance['% Исходящих'].idxmax()
        best_outgoing_manager = df_performance.iloc[best_outgoing_idx]
        st.info(f"📱 **Больше всего исходящих:** {best_outgoing_manager['Менеджер']} ({best_outgoing_manager['% Исходящих']}%)")

