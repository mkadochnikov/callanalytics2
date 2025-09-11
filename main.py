#!/usr/bin/env python3
"""
Streamlit веб-интерфейс для системы анализа звонков с аналитикой по менеджерам и возражениям
БЕЗ АНАЛИЗА ТОНАЛЬНОСТИ
"""

from config import *
from main_analyzer import BitrixCallAnalyzer
from manager_analytics import show_manager_analytics
from pathlib import Path
import datetime
import streamlit as st
import pandas as pd
import pytz


def apply_professional_theme():
    """Применяет профессиональную тему оформления"""
    st.markdown("""
    <style>
    /* Основные настройки */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Заголовки */
    h1, h2, h3 {
        font-weight: 600;
        color: #1a1a1a;
        letter-spacing: -0.02em;
    }

    /* Метрики */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e1e4e8;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }

    [data-testid="metric-container"] label {
        font-size: 12px;
        font-weight: 500;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 24px;
        font-weight: 700;
        color: #1a1a1a;
    }

    /* Кнопки */
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 13px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

    /* Sidebar улучшения */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }

    /* Стили для таблицы из колонок */
    .table-header {
        background: #2c3e50;
        color: white;
        padding: 12px 8px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 0.5px;
        text-align: center;
        margin-bottom: 0;
    }

    .table-row {
        border-bottom: 1px solid #e9ecef;
        padding: 8px;
        transition: background-color 0.2s ease;
    }

    .table-row:hover {
        background-color: #f8f9fa;
    }

    .table-cell {
        font-size: 13px;
        color: #2c3e50;
        display: flex;
        align-items: center;
        min-height: 35px;
    }

    .table-cell-center {
        justify-content: center;
        text-align: center;
    }

    /* Стили для модального окна */
    .info-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #3498db;
        margin-bottom: 15px;
    }

    .warning-card {
        background: #fff3cd;
        border-left-color: #ffc107;
    }

    .success-card {
        background: #d1ecf1;
        border-left-color: #17a2b8;
    }

    /* Кнопка-лупа */
    .view-button {
        background: none !important;
        border: none !important;
        font-size: 18px !important;
        padding: 4px 8px !important;
        border-radius: 4px !important;
        color: #3498db !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }

    .view-button:hover {
        background: #e8f4f8 !important;
        transform: scale(1.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Основная функция со Streamlit интерфейсом"""
    st.set_page_config(
        page_title="Bitrix24 AI Аналитика",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Применяем профессиональную тему
    apply_professional_theme()

    # Инициализация session state
    if 'show_call_modal' not in st.session_state:
        st.session_state.show_call_modal = False

    # Заголовок с премиум-стилем
    st.markdown("""
    <div style='text-align: center; padding: 20px 0; margin-bottom: 30px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.15);'>
        <h1 style='color: white; margin: 0; font-size: 32px; font-weight: 700;'>
            📊 Платформа AI-аналитики Bitrix24
        </h1>
        <p style='color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 14px; font-weight: 400;'>
            Корпоративная система анализа звонков • Локальные AI модели
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Проверяем доступность CUDA
    import torch
    device_info = "🚀 NVIDIA CUDA GPU" if torch.cuda.is_available() else "💻 Режим CPU"
    st.sidebar.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 10px; border-radius: 8px; text-align: center;
                font-weight: 600; font-size: 13px; margin-bottom: 20px;'>
        {device_info}
    </div>
    """, unsafe_allow_html=True)

    # Инициализация analyzer
    if 'analyzer' not in st.session_state:
        with st.spinner("🔄 Инициализация AI моделей..."):
            st.session_state.analyzer = BitrixCallAnalyzer()

    analyzer = st.session_state.analyzer

    # Sidebar с настройками
    with st.sidebar:
        st.markdown("### ⚙️ Панель настроек")

        # Настройки Bitrix24
        with st.expander("🏢 Подключение Bitrix24", expanded=False):
            webhook_url = st.text_input("Webhook URL", value=analyzer.bitrix_api.webhook_url, type="password")
            username = st.text_input("Имя пользователя", value=analyzer.bitrix_api.username)
            password = st.text_input("Пароль", type="password", value=analyzer.bitrix_api.password)

            if st.button("💾 Сохранить настройки"):
                analyzer.bitrix_api.webhook_url = webhook_url
                analyzer.bitrix_api.username = username
                analyzer.bitrix_api.password = password
                st.success("✅ Настройки сохранены")

        # Информация о моделях
        with st.expander("🤖 Статус AI моделей", expanded=True):
            whisper_status = "✅ Активна" if analyzer.audio_processor.whisper_model else "❌ Ошибка"
            classifier_status = "✅ Активен" if analyzer.ai_analyzer.classifier else "❌ Ошибка"

            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 10px; border-radius: 8px;'>
                <div style='margin-bottom: 8px;'>
                    <span style='font-weight: 600; font-size: 12px;'>Whisper ASR:</span>
                    <span style='float: right; font-size: 12px;'>{whisper_status}</span>
                </div>
                <div>
                    <span style='font-weight: 600; font-size: 12px;'>Классификатор:</span>
                    <span style='float: right; font-size: 12px;'>{classifier_status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Статистика по возражениям
        if hasattr(analyzer.ai_analyzer, 'custom_objections'):
            from config import OBJECTION_CATEGORIES
            custom_count = len(analyzer.ai_analyzer.custom_objections)
            total_count = len(OBJECTION_CATEGORIES) + custom_count

            st.markdown(f"""
            <div style='background: #e8f4f8; padding: 10px; border-radius: 8px; margin-top: 10px;'>
                <div style='font-weight: 600; font-size: 12px; color: #2c3e50; margin-bottom: 5px;'>
                    Категории возражений
                </div>
                <div style='font-size: 11px; color: #5a6c7d;'>
                    Всего: {total_count} | Базовых: {len(OBJECTION_CATEGORIES)} | Пользовательских: {custom_count}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Основной интерфейс
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📅 Выбор периода анализа")

        # Выбор даты
        date_option = st.radio(
            "",
            ["Один день", "Диапазон дат"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if date_option == "Один день":
            selected_date = st.date_input(
                "Выберите дату",
                value=datetime.date.today() - datetime.timedelta(days=1)
            )
            start_date = end_date = selected_date
        else:
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input(
                    "Начальная дата",
                    value=datetime.date.today() - datetime.timedelta(days=7)
                )
            with col_end:
                end_date = st.date_input(
                    "Конечная дата",
                    value=datetime.date.today() - datetime.timedelta(days=1)
                )

    with col2:
        st.markdown("### 🚀 Действия")

        # Проверяем доступность переанализа
        reanalysis_available = False
        total_cached_calls = 0

        if start_date <= end_date:
            date_range_check = pd.date_range(start_date, end_date)
            for date_check in date_range_check:
                target_datetime = datetime.datetime.combine(date_check.date(), datetime.time())
                reanalysis_info = analyzer.data_manager.get_reanalysis_info(target_datetime)
                if reanalysis_info["can_reanalyze"]:
                    reanalysis_available = True
                    total_cached_calls += reanalysis_info.get("calls_with_transcripts", 0)

        # Опция принудительного переанализа
        force_reanalyze = False
        if reanalysis_available:
            force_reanalyze = st.checkbox(
                f"🔄 Переанализировать данные ({total_cached_calls} звонков)",
                help="Обновить анализ возражений для уже транскрибированных звонков"
            )

        # Основная кнопка обработки
        button_text = "🔄 Переанализировать" if force_reanalyze else "🤖 Загрузить и проанализировать"

        if st.button(button_text, type="primary", use_container_width=True):
            if start_date <= end_date:
                processing_text = "Переанализ с обновленными алгоритмами..." if force_reanalyze else "Обработка с AI моделями..."

                with st.spinner(processing_text):
                    all_calls = []
                    date_range = pd.date_range(start_date, end_date)
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, date in enumerate(date_range):
                        action_text = "Переанализ" if force_reanalyze else "Обработка"
                        status_text.info(f"🔄 {action_text}: {date.strftime('%d.%m.%Y')} ({i + 1}/{len(date_range)})")

                        target_datetime = datetime.datetime.combine(date.date(), datetime.time())
                        calls = analyzer.process_calls_for_date(target_datetime, force_reanalyze=force_reanalyze)
                        all_calls.extend(calls)
                        progress_bar.progress((i + 1) / len(date_range))

                    st.session_state.all_calls = all_calls
                    status_text.success(f"✅ Успешно обработано {len(all_calls)} звонков")
            else:
                st.error("❌ Начальная дата должна быть раньше конечной")

        # Кнопка экспорта
        if 'all_calls' in st.session_state and st.session_state.all_calls:
            if st.button("📄 Экспорт отчета", width='stretch'):
                report_path = analyzer.export_objections_report(st.session_state.all_calls)
                st.success(f"✅ Отчет сохранен: {report_path}")

    # Показываем результаты анализа
    show_analysis_results()


def show_analysis_results():
    """Показывает результаты анализа"""
    if 'all_calls' not in st.session_state or not st.session_state.all_calls:
        return

    st.markdown("### 📊 Панель аналитики")

    calls_data = st.session_state.all_calls
    analyzer = st.session_state.analyzer
    objections_stats = analyzer.get_objections_statistics(calls_data)

    # Метрики в премиум-стиле
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Всего звонков", len(calls_data))

    with col2:
        incoming = sum(1 for call in calls_data if call.get('call_direction') == 'incoming')
        st.metric("Входящие", incoming)

    with col3:
        outgoing = sum(1 for call in calls_data if call.get('call_direction') == 'outgoing')
        st.metric("Исходящие", outgoing)

    with col4:
        with_audio = sum(1 for call in calls_data if 'audio_file' in call)
        st.metric("С аудио", with_audio)

    with col5:
        analyzed = sum(1 for call in calls_data if 'analysis' in call)
        st.metric("Проанализировано", analyzed)

    with col6:
        objections_count = objections_stats['total_calls_with_objections']
        objections_percent = (objections_count / len(calls_data) * 100) if calls_data else 0
        st.metric("Возражения", f"{objections_count} ({objections_percent:.1f}%)")

    st.markdown("---")

    # Аналитика по менеджерам
    show_manager_analytics(calls_data)

    st.markdown("---")

    # Сводные таблицы
    show_summary_tables(calls_data, objections_stats)

    # Детализация
    show_call_details(calls_data)


def show_summary_tables(calls_data, objections_stats):
    """Показывает сводные таблицы"""
    st.markdown("### 📈 Сводка AI-анализа")

    col1, col2 = st.columns(2)

    with col1:
        topic_data = {}
        for call in calls_data:
            analysis = call.get('analysis', {})
            topic = analysis.get('topic', 'Не определена')
            topic_data[topic] = topic_data.get(topic, 0) + 1

        if topic_data:
            st.markdown("#### 🎯 Темы звонков")
            topic_df = pd.DataFrame([
                {'Тема': topic, 'Количество': count, 'Доля': f"{count / len(calls_data) * 100:.1f}%"}
                for topic, count in sorted(topic_data.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(topic_df, use_container_width=True, hide_index=True)

    with col2:
        objections = objections_stats['objections']
        if objections:
            st.markdown("#### 🚫 Возражения клиентов")
            objection_df = pd.DataFrame([
                {'Возражение': objection, 'Количество': count,
                 'Доля': f"{count / objections_stats['total_calls_with_objections'] * 100:.1f}%"}
                for objection, count in sorted(objections.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(objection_df, width='stretch', hide_index=True)


def show_call_details(calls_data):
    """Показывает детализацию звонков с кликабельными иконками лупы"""
    st.header("🔍 Детализация звонков с ИИ анализом")

    if not calls_data:
        st.info("Нет данных для отображения")
        return

    # Заголовки таблицы
    header_cols = st.columns([2, 2, 2, 1, 1.5, 3, 4, 1])
    headers = ["Дата/время", "Менеджер", "Телефон", "Тип", "Длительность", "Тема (ИИ)", "Возражение → Рекомендация",
               "Просмотр"]

    for col, header in zip(header_cols, headers):
        with col:
            st.markdown(f'<div class="table-header">{header}</div>', unsafe_allow_html=True)

    # Строки данных
    for i, call in enumerate(calls_data):
        analysis = call.get('analysis', {})

        # Форматируем время
        call_time = call.get('CALL_START_DATE', '')
        if call_time and 'T' in call_time:
            try:
                dt = datetime.datetime.fromisoformat(call_time.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%d.%m.%Y %H:%M')
            except:
                formatted_time = call_time
        else:
            formatted_time = call_time

        # Определяем тип звонка со стрелочками
        call_direction = call.get('call_direction', '')
        if call_direction == 'incoming':
            call_type = '←'
        elif call_direction == 'outgoing':
            call_type = '→'
        else:
            call_type = '•'

        # Получаем данные о возражении
        objection_reason = analysis.get('objection_reason', '')
        objection_recommendation = analysis.get('objection_recommendation', '')

        objection_summary = ""
        if objection_reason:
            if len(objection_reason) > 50:
                objection_summary = objection_reason[:47] + "..."
            else:
                objection_summary = objection_reason

            if objection_recommendation:
                if len(objection_recommendation) > 35:
                    rec_summary = objection_recommendation[:32] + "..."
                else:
                    rec_summary = objection_recommendation
                objection_summary += f" → {rec_summary}"
        else:
            objection_summary = "—"

        # Проверяем доступность данных
        has_data = bool(call.get('transcript') or call.get('audio_file'))

        # Создаем строку таблицы
        with st.container():
            row_cols = st.columns([2, 2, 2, 1, 1.5, 3, 4, 1])

            with row_cols[0]:
                st.markdown(f'<div class="table-cell">{formatted_time}</div>', unsafe_allow_html=True)
            with row_cols[1]:
                st.markdown(f'<div class="table-cell">{call.get("user_name", "")}</div>', unsafe_allow_html=True)
            with row_cols[2]:
                st.markdown(f'<div class="table-cell">{call.get("PHONE_NUMBER", "")}</div>', unsafe_allow_html=True)
            with row_cols[3]:
                st.markdown(f'<div class="table-cell table-cell-center">{call_type}</div>', unsafe_allow_html=True)
            with row_cols[4]:
                st.markdown(f'<div class="table-cell">{call.get("CALL_DURATION", 0)} сек</div>', unsafe_allow_html=True)
            with row_cols[5]:
                st.markdown(f'<div class="table-cell">{analysis.get("topic", "Неопределенная тема")}</div>',
                            unsafe_allow_html=True)
            with row_cols[6]:
                st.markdown(f'<div class="table-cell">{objection_summary}</div>', unsafe_allow_html=True)
            with row_cols[7]:
                if has_data:
                    if st.button("🔍", key=f"view_{i}", help="Просмотр деталей", type="secondary"):
                        st.session_state.show_call_modal = True
                        st.session_state.selected_call_data = call
                        st.rerun()
                else:
                    st.markdown('<div class="table-cell table-cell-center">—</div>', unsafe_allow_html=True)

    # Модальное окно
    if st.session_state.get('show_call_modal', False) and 'selected_call_data' in st.session_state:
        show_call_modal_wide(st.session_state.selected_call_data)


def show_call_modal_wide(call_data):
    """Показывает широкое модальное окно с деталями звонка"""

    # Проверяем доступность st.dialog
    try:
        @st.dialog("📞 Детали звонка", width="large")
        def call_modal_dialog():
            show_call_details_content(call_data)

        call_modal_dialog()
    except:
        # Fallback для старых версий Streamlit
        with st.container():
            st.markdown("---")
            st.markdown("## 📞 Детали звонка")
            show_call_details_content(call_data)

            if st.button("❌ Закрыть", key="close_modal"):
                st.session_state.show_call_modal = False
                st.rerun()


def show_call_details_content(call_data):
    """Содержимое модального окна с деталями звонка"""

    # Основная информация о звонке
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📞 Информация о звонке**")

        # Форматируем время
        call_time = call_data.get('CALL_START_DATE', '')
        if call_time and 'T' in call_time:
            try:
                dt = datetime.datetime.fromisoformat(call_time.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%d.%m.%Y в %H:%M')
            except:
                formatted_time = call_time
        else:
            formatted_time = call_time

        direction = call_data.get('call_direction', '')
        direction_text = 'Входящий ←' if direction == 'incoming' else 'Исходящий →' if direction == 'outgoing' else 'Неизвестно'

        st.markdown(f"""
        <div class="info-card">
            <strong>Дата:</strong> {formatted_time}<br>
            <strong>Менеджер:</strong> {call_data.get('user_name', 'Неизвестный')}<br>
            <strong>Телефон:</strong> {call_data.get('PHONE_NUMBER', 'Неизвестный')}<br>
            <strong>Длительность:</strong> {call_data.get('CALL_DURATION', 0)} секунд<br>
            <strong>Направление:</strong> {direction_text}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**🤖 ИИ Анализ**")

        analysis = call_data.get('analysis', {})
        topic = analysis.get('topic', 'Не определена')
        objection = analysis.get('objection_reason', '')
        recommendation = analysis.get('objection_recommendation', '')

        if objection:
            st.markdown(f"""
            <div class="info-card warning-card">
                <strong>Тема:</strong> {topic}<br>
                <strong>Возражение:</strong> {objection}<br>
                <strong>Рекомендация:</strong> {recommendation if recommendation else 'Не предоставлена'}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-card success-card">
                <strong>Тема:</strong> {topic}<br>
                <strong>Возражения:</strong> Не обнаружены<br>
                <strong>Статус:</strong> Позитивный звонок
            </div>
            """, unsafe_allow_html=True)

    # Табы для аудио и транскрипции
    tab1, tab2 = st.tabs(["🔊 Аудиозапись", "📝 Транскрипция"])

    with tab1:
        audio_file_path = call_data.get('audio_file')
        if audio_file_path and Path(audio_file_path).exists():
            try:
                with open(audio_file_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()

                st.audio(audio_bytes, format='audio/mp3')

                # Информация о файле
                file_size = Path(audio_file_path).stat().st_size / (1024 * 1024)
                st.caption(f"Размер файла: {file_size:.2f} МБ")

            except Exception as e:
                st.error(f"Ошибка при загрузке аудио: {str(e)}")
        else:
            st.info("Аудиозапись недоступна")

    with tab2:
        transcript = call_data.get('transcript')
        if transcript:
            st.text_area(
                "Текст разговора",
                value=transcript,
                height=400,
                label_visibility="collapsed"
            )
        else:
            st.info("Транскрипция недоступна")


if __name__ == "__main__":
    main()