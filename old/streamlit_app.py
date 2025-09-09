#!/usr/bin/env python3
"""
Streamlit веб-интерфейс для системы анализа звонков
"""

from config import *
from main_analyzer import BitrixCallAnalyzer


def main():
    """Основная функция со Streamlit интерфейсом"""
    st.set_page_config(
        page_title="Bitrix24 Local AI Analytics",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 Bitrix24 Local AI Analytics")
    st.markdown("**Полностью локальная система анализа звонков с ИИ** • Whisper + Transformers")

    # Проверяем доступность CUDA
    device_info = "🔥 CUDA GPU" if torch.cuda.is_available() else "💻 CPU"
    st.sidebar.info(f"Устройство обработки: {device_info}")

    # Инициализация analyzer
    if 'analyzer' not in st.session_state:
        with st.spinner("Загрузка моделей ИИ..."):
            st.session_state.analyzer = BitrixCallAnalyzer()

    analyzer = st.session_state.analyzer

    # Sidebar с настройками
    with st.sidebar:
        st.header("⚙️ Настройки подключения")

        # Настройки Bitrix24
        st.subheader("🏢 Bitrix24")
        webhook_url = st.text_input("Webhook URL", value=analyzer.bitrix_api.webhook_url, type="password")
        username = st.text_input("Username", value=analyzer.bitrix_api.username)
        password = st.text_input("Password", type="password", value=analyzer.bitrix_api.password)

        if st.button("💾 Сохранить настройки"):
            # Обновляем настройки
            analyzer.bitrix_api.webhook_url = webhook_url
            analyzer.bitrix_api.username = username
            analyzer.bitrix_api.password = password
            st.success("✅ Настройки сохранены!")

        st.markdown("---")

        # Информация о моделях
        st.subheader("🤖 Статус локальных моделей")

        whisper_status = "✅ Загружена" if analyzer.audio_processor.whisper_model else "❌ Ошибка"
        st.write(f"**Whisper:** {whisper_status}")

        classifier_status = "✅ Загружена" if analyzer.ai_analyzer.classifier else "❌ Ошибка"
        st.write(f"**Классификация тем:** {classifier_status}")

    # Основной интерфейс
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📅 Выбор периода анализа")

        # Выбор даты
        date_option = st.radio(
            "Выберите период:",
            ["Один день", "Диапазон дат"],
            horizontal=True
        )

        if date_option == "Один день":
            selected_date = st.date_input(
                "Выберите дату:",
                value=datetime.date.today() - datetime.timedelta(days=1)
            )
            start_date = end_date = selected_date
        else:
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input(
                    "Начальная дата:",
                    value=datetime.date.today() - datetime.timedelta(days=7)
                )
            with col_end:
                end_date = st.date_input(
                    "Конечная дата:",
                    value=datetime.date.today() - datetime.timedelta(days=1)
                )

    with col2:
        st.header("🚀 Действия")

        if st.button("🤖 Загрузить и проанализировать", type="primary", use_container_width=True):
            if start_date <= end_date:
                with st.spinner("Обработка с локальными ИИ моделями..."):
                    all_calls = []

                    # Прогресс бар
                    date_range = pd.date_range(start_date, end_date)
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, date in enumerate(date_range):
                        status_text.info(
                            f"🔄 Обработка: {date.strftime('%d.%m.%Y')} (день {i + 1} из {len(date_range)})")

                        target_datetime = datetime.datetime.combine(date.date(), datetime.time())
                        calls = analyzer.process_calls_for_date(target_datetime)
                        all_calls.extend(calls)

                        progress_bar.progress((i + 1) / len(date_range))

                    st.session_state.all_calls = all_calls
                    status_text.success(f"✅ Обработано {len(all_calls)} звонков с помощью локальных ИИ моделей")
            else:
                st.error("❌ Начальная дата должна быть меньше конечной")

    # Показываем результаты анализа
    show_analysis_results()


def show_analysis_results():
    """Показывает результаты анализа"""
    if 'all_calls' not in st.session_state or not st.session_state.all_calls:
        return

    st.header("📊 Статистика по обработанным данным")

    calls_data = st.session_state.all_calls

    # Метрики
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
        unknown = sum(1 for call in calls_data if call.get('call_direction') == 'unknown')
        st.metric("Неопределенные", unknown)

    with col5:
        with_audio = sum(1 for call in calls_data if 'audio_file' in call)
        st.metric("С аудиозаписью", with_audio)

    with col6:
        analyzed = sum(1 for call in calls_data if 'analysis' in call)
        st.metric("Проанализировано ИИ", analyzed)

    # Сводные таблицы
    show_summary_tables(calls_data)

    # Детализация
    show_call_details(calls_data)


def show_summary_tables(calls_data):
    """Показывает сводные таблицы"""
    st.header("📊 Сводные таблицы ИИ анализа")

    col1, col2 = st.columns(2)

    with col1:
        # Таблица тем
        topic_data = {}
        for call in calls_data:
            analysis = call.get('analysis', {})
            topic = analysis.get('topic', 'Неопределенная тема')
            topic_data[topic] = topic_data.get(topic, 0) + 1

        if topic_data:
            st.subheader("🎯 Темы звонков")
            topic_df = pd.DataFrame([
                {'Тема': topic, 'Количество': count, 'Процент': f"{count / len(calls_data) * 100:.1f}%"}
                for topic, count in sorted(topic_data.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(topic_df, use_container_width=True, hide_index=True)

    with col2:
        # Таблица отказов
        rejection_data = {}
        for call in calls_data:
            analysis = call.get('analysis', {})
            rejection = analysis.get('rejection_reason')
            if rejection:
                rejection_data[rejection] = rejection_data.get(rejection, 0) + 1

        if rejection_data:
            st.subheader("❌ Причины отказов")
            rejection_df = pd.DataFrame([
                {'Причина отказа': reason, 'Количество': count,
                 'Процент': f"{count / len([c for c in calls_data if c.get('analysis', {}).get('rejection_reason')]) * 100:.1f}%"}
                for reason, count in sorted(rejection_data.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(rejection_df, use_container_width=True, hide_index=True)
        else:
            st.subheader("❌ Причины отказов")
            st.info("Отказы не обнаружены в анализируемых звонках")


def show_call_details(calls_data):
    """Показывает детализацию звонков"""
    st.header("🔍 Детализация звонков с ИИ анализом")

    # Подготавливаем данные для таблицы
    table_data = []
    for call in calls_data:
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

        table_data.append({
            'Дата/время': formatted_time,
            'Менеджер': call.get('user_name', ''),
            'Телефон': call.get('PHONE_NUMBER', ''),
            'Тип': call.get('call_direction', 'unknown').replace('incoming', '📞 Входящий').replace('outgoing',
                                                                                                   '📱 Исходящий').replace(
                'unknown', '❓ Неопределенный'),
            'Длительность': f"{call.get('CALL_DURATION', 0)} сек",
            'Тема (ИИ)': analysis.get('topic', 'Неопределенная тема'),
            'Причина отказа (ИИ)': analysis.get('rejection_reason', ''),
            'Транскрипция': '✅ Есть' if call.get('transcript') else '❌ Нет'
        })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

        # Показываем примеры транскрипций
        show_transcript_examples(calls_data)


def show_transcript_examples(calls_data):
    """Показывает примеры транскрипций"""
    st.subheader("📝 Примеры транскрипций и анализа")

    calls_with_transcripts = [call for call in calls_data if call.get('transcript') and call.get('analysis')]

    if calls_with_transcripts:
        selected_call = st.selectbox(
            "Выберите звонок для просмотра:",
            range(min(10, len(calls_with_transcripts))),
            format_func=lambda
                x: f"Звонок {x + 1}: {calls_with_transcripts[x].get('user_name', 'Unknown')} - {calls_with_transcripts[x]['analysis'].get('topic', 'Неопределенная тема')}"
        )

        if selected_call is not None:
            call = calls_with_transcripts[selected_call]
            analysis = call.get('analysis', {})

            col1, col2 = st.columns([2, 1])

            with col1:
                st.write("**Транскрипция:**")
                st.text_area("Текст разговора", call.get('transcript', ''), height=200, disabled=True,
                             label_visibility="collapsed")

            with col2:
                st.write("**ИИ Анализ:**")
                st.write(f"**Тема:** {analysis.get('topic', 'Неопределенная тема')}")

                if analysis.get('rejection_reason'):
                    st.write(f"**Причина отказа:** {analysis['rejection_reason']}")
                else:
                    st.write("**Причина отказа:** Не выявлена")

                key_points = analysis.get('key_points', [])
                if key_points:
                    st.write("**Ключевые моменты:**")
                    for i, point in enumerate(key_points, 1):
                        st.write(f"{i}. {point}")
                else:
                    st.write("**Ключевые моменты:** Не выявлены")


if __name__ == "__main__":
    main()