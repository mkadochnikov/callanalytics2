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
    import torch
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

        # Информация о моделях (БЕЗ sentiment)
        st.subheader("🤖 Статус локальных моделей")

        whisper_status = "✅ Загружена" if analyzer.audio_processor.whisper_model else "❌ Ошибка"
        st.write(f"**Whisper:** {whisper_status}")

        classifier_status = "✅ Загружена" if analyzer.ai_analyzer.classifier else "❌ Ошибка"
        st.write(f"**Классификация тем:** {classifier_status}")

        # Статистика по возражениям (БЕЗ sentiment)
        if hasattr(analyzer.ai_analyzer, 'custom_objections'):
            from config import OBJECTION_CATEGORIES
            custom_count = len(analyzer.ai_analyzer.custom_objections)
            total_count = len(OBJECTION_CATEGORIES) + custom_count
            st.write(
                f"**Категории возражений:** {total_count} (базовых: {len(OBJECTION_CATEGORIES)}, пользовательских: {custom_count})")

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

        # Проверяем доступность переанализа
        reanalysis_available = False
        total_cached_calls = 0

        if start_date <= end_date:
            # Проверяем для каждой даты в диапазоне
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
                f"🔄 Переанализировать существующие данные ({total_cached_calls} звонков)",
                help="Обновить анализ возражений для уже транскрибированных звонков без повторного скачивания аудио"
            )

            if force_reanalyze:
                st.info(
                    "📝 Будет выполнен только переанализ текстов. Аудиозаписи и транскрипции останутся без изменений.")

        # Основная кнопка обработки
        button_text = "🔄 Переанализировать" if force_reanalyze else "🤖 Загрузить и проанализировать"

        if st.button(button_text, type="primary", use_container_width=True):
            if start_date <= end_date:
                processing_text = "Переанализ с обновленными алгоритмами..." if force_reanalyze else "Обработка с локальными ИИ моделями..."

                with st.spinner(processing_text):
                    all_calls = []

                    # Прогресс бар
                    date_range = pd.date_range(start_date, end_date)
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, date in enumerate(date_range):
                        action_text = "Переанализ" if force_reanalyze else "Обработка"
                        status_text.info(
                            f"🔄 {action_text}: {date.strftime('%d.%m.%Y')} (день {i + 1} из {len(date_range)})")

                        target_datetime = datetime.datetime.combine(date.date(), datetime.time())

                        # Передаем флаг принудительного переанализа
                        calls = analyzer.process_calls_for_date(target_datetime, force_reanalyze=force_reanalyze)
                        all_calls.extend(calls)

                        progress_bar.progress((i + 1) / len(date_range))

                    st.session_state.all_calls = all_calls

                    if force_reanalyze:
                        status_text.success(f"✅ Переанализировано {len(all_calls)} звонков с обновленными алгоритмами")
                    else:
                        status_text.success(f"✅ Обработано {len(all_calls)} звонков с помощью локальных ИИ моделей")
            else:
                st.error("❌ Начальная дата должна быть меньше конечной")

        # Кнопка экспорта отчета по возражениям
        if 'all_calls' in st.session_state and st.session_state.all_calls:
            if st.button("📄 Экспорт отчета по возражениям", use_container_width=True):
                report_path = analyzer.export_objections_report(st.session_state.all_calls)
                st.success(f"✅ Отчет сохранен: {report_path}")

    # Показываем результаты анализа
    show_analysis_results()


def show_analysis_results():
    """Показывает результаты анализа БЕЗ ТОНАЛЬНОСТИ"""
    if 'all_calls' not in st.session_state or not st.session_state.all_calls:
        return

    st.header("📊 Статистика по обработанным данным")

    calls_data = st.session_state.all_calls

    # Получаем статистику возражений (БЕЗ тональности)
    analyzer = st.session_state.analyzer
    objections_stats = analyzer.get_objections_statistics(calls_data)

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
        with_audio = sum(1 for call in calls_data if 'audio_file' in call)
        st.metric("С аудиозаписью", with_audio)

    with col5:
        analyzed = sum(1 for call in calls_data if 'analysis' in call)
        st.metric("Проанализировано ИИ", analyzed)

    with col6:
        objections_count = objections_stats['total_calls_with_objections']
        objections_percent = (objections_count / len(calls_data) * 100) if calls_data else 0
        st.metric("Возражения", f"{objections_count} ({objections_percent:.1f}%)")

    # РАЗДЕЛ: Аналитика по менеджерам
    st.markdown("---")
    show_manager_analytics(calls_data)

    # Разделитель перед остальными разделами
    st.markdown("---")

    # Сводные таблицы (БЕЗ тональности)
    show_summary_tables(calls_data, objections_stats)

    # Детализация (БЕЗ тональности)
    show_call_details(calls_data)


def show_objections_analysis(calls_data, objections_stats):
    """Показывает детальный анализ возражений БЕЗ ТОНАЛЬНОСТИ"""
    st.header("🚫 Анализ возражений клиентов")

    if not objections_stats['objections']:
        st.info("В данном периоде возражения не обнаружены")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Топ возражений")

        # Создаем DataFrame для визуализации
        objections_df = pd.DataFrame([
            {
                'Возражение': objection.replace('🎯 ', '').replace('⏰ ', '').replace('🔄 ', '').replace('🔍 ', '').replace(
                    '🤝 ', '').replace('⚙️ ', '').replace('🛡️ ', '').replace('⭐ ', '').replace('🔧 ', '').replace('🎓 ',
                                                                                                                '').replace(
                    '📝 ', '').replace('❓ ', ''),
                'Количество': count,
                'Процент': f"{count / objections_stats['total_calls_with_objections'] * 100:.1f}%",
                'Оригинальное название': objection
            }
            for objection, count in sorted(objections_stats['objections'].items(), key=lambda x: x[1], reverse=True)
        ])

        st.dataframe(objections_df[['Возражение', 'Количество', 'Процент']], use_container_width=True, hide_index=True)

    with col2:
        st.subheader("💡 Рекомендации по работе")

        # Показываем рекомендации для топ-5 возражений
        top_objections = list(objections_stats['objections'].keys())[:5]
        objection_categories = objections_stats['objection_categories']

        for objection in top_objections:
            if objection in objection_categories:
                recommendation = objection_categories[objection].get('recommendation', 'Индивидуальный подход')
                st.write(f"**{objection}**")
                st.write(f"→ {recommendation}")
                st.write("")


def show_summary_tables(calls_data, objections_stats):
    """Показывает сводные таблицы БЕЗ ТОНАЛЬНОСТИ"""
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
        # Таблица возражений
        objections = objections_stats['objections']
        if objections:
            st.subheader("🚫 Возражения клиентов")
            objection_df = pd.DataFrame([
                {
                    'Возражение': objection,
                    'Количество': count,
                    'Процент': f"{count / objections_stats['total_calls_with_objections'] * 100:.1f}%"
                }
                for objection, count in sorted(objections.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(objection_df, use_container_width=True, hide_index=True)
        else:
            st.subheader("🚫 Возражения клиентов")
            st.info("Возражения не обнаружены в анализируемых звонках")


def show_call_details(calls_data):
    """Показывает детализацию звонков с возможностью просмотра транскрипций и прослушивания аудио"""
    st.header("🔍 Детализация звонков с ИИ анализом")

    # Подготавливаем данные для таблицы
    table_data = []
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

        # Получаем данные о возражении
        objection_reason = analysis.get('objection_reason', '')
        objection_recommendation = analysis.get('objection_recommendation', '')

        # Формируем строку с возражением и рекомендацией
        objection_info = ""
        if objection_reason:
            objection_info = f"{objection_reason}"
            if objection_recommendation:
                objection_info += f" → {objection_recommendation}"

        table_data.append({
            'index': i,
            'Дата/время': formatted_time,
            'Менеджер': call.get('user_name', ''),
            'Телефон': call.get('PHONE_NUMBER', ''),
            'Тип': '📞 Входящий' if call.get('call_direction') == 'incoming' else '📱 Исходящий' if call.get(
                'call_direction') == 'outgoing' else '❓ Неопределенный',
            'Длительность': f"{call.get('CALL_DURATION', 0)} сек",
            'Тема (ИИ)': analysis.get('topic', 'Неопределенная тема'),
            'Возражение → Рекомендация': objection_info,
            'has_transcript': bool(call.get('transcript')),
            'has_audio': bool(call.get('audio_file'))
        })

    if table_data:
        # Создаем DataFrame
        df = pd.DataFrame(table_data)

        # Создаем колонки для кнопок
        col_headers = st.columns([1, 1.5, 1.5, 1, 1, 1.5, 3, 0.8, 0.8])
        headers = ['Дата/время', 'Менеджер', 'Телефон', 'Тип', 'Длительность', 'Тема (ИИ)', 'Возражение → Рекомендация',
                   'Транскрипция', 'Аудио']

        for col, header in zip(col_headers, headers):
            col.markdown(f"**{header}**")

        # Отображаем каждую строку с кнопками
        for idx, row in df.iterrows():
            cols = st.columns([1, 1.5, 1.5, 1, 1, 1.5, 3, 0.8, 0.8])

            # Данные
            cols[0].write(row['Дата/время'])
            cols[1].write(row['Менеджер'])
            cols[2].write(row['Телефон'])
            cols[3].write(row['Тип'])
            cols[4].write(row['Длительность'])
            cols[5].write(row['Тема (ИИ)'])
            cols[6].write(row['Возражение → Рекомендация'])

            # Кнопка транскрипции
            if row['has_transcript']:
                if cols[7].button('📄', key=f"transcript_{row['index']}", help="Прочитать транскрипцию"):
                    show_transcript_modal(calls_data[row['index']])
            else:
                cols[7].write('❌')

            # Кнопка аудио
            if row['has_audio']:
                if cols[8].button('🔊', key=f"audio_{row['index']}", help="Прослушать запись"):
                    show_audio_modal(calls_data[row['index']])
            else:
                cols[8].write('❌')

        # Статистика под таблицей
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_with_transcript = sum(1 for d in table_data if d['has_transcript'])
            st.metric("С транскрипцией", f"{total_with_transcript}/{len(table_data)}")

        with col2:
            total_with_audio = sum(1 for d in table_data if d['has_audio'])
            st.metric("С аудиозаписью", f"{total_with_audio}/{len(table_data)}")

        with col3:
            total_with_objections = sum(1 for d in table_data if d['Возражение → Рекомендация'])
            st.metric("С возражениями", f"{total_with_objections}/{len(table_data)}")

        with col4:
            avg_duration = df['Длительность'].apply(lambda x: int(x.split()[0])).mean()
            st.metric("Средняя длительность", f"{avg_duration:.0f} сек")


def show_transcript_modal(call_data):
    """Показывает модальное окно с транскрипцией"""
    # Получаем информацию о звонке
    call_info = f"""
    **Менеджер:** {call_data.get('user_name', 'Неизвестный')}  
    **Телефон:** {call_data.get('PHONE_NUMBER', 'Неизвестный')}  
    **Длительность:** {call_data.get('CALL_DURATION', 0)} сек
    """

    # Получаем анализ
    analysis = call_data.get('analysis', {})
    analysis_info = f"""
    **Тема:** {analysis.get('topic', 'Не определена')}  
    **Возражение:** {analysis.get('objection_reason', 'Не обнаружено')}  
    **Рекомендация:** {analysis.get('objection_recommendation', 'Нет')}
    """

    # Создаем модальное окно с использованием expander
    with st.expander("📄 Транскрипция разговора", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Информация о звонке")
            st.markdown(call_info)

        with col2:
            st.markdown("### ИИ Анализ")
            st.markdown(analysis_info)

        st.markdown("---")
        st.markdown("### Текст разговора")

        # Отображаем транскрипцию
        transcript = call_data.get('transcript', 'Транскрипция отсутствует')

        # Форматируем текст для лучшей читаемости
        formatted_transcript = transcript.replace('. ', '.\n\n')

        # Используем text_area для возможности копирования
        st.text_area(
            "Транскрипция",
            formatted_transcript,
            height=400,
            disabled=True,
            label_visibility="collapsed"
        )

        # Кнопка копирования
        if st.button("📋 Копировать текст", key=f"copy_{call_data.get('ID', '')}"):
            st.code(transcript)
            st.info("Выделите и скопируйте текст выше")


def show_audio_modal(call_data):
    """Показывает модальное окно с аудиоплеером"""
    audio_file_path = call_data.get('audio_file')

    if audio_file_path and Path(audio_file_path).exists():
        # Создаем модальное окно с использованием expander
        with st.expander("🔊 Прослушивание записи", expanded=True):
            # Информация о звонке
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Менеджер", call_data.get('user_name', 'Неизвестный'))

            with col2:
                st.metric("Телефон", call_data.get('PHONE_NUMBER', 'Неизвестный'))

            with col3:
                st.metric("Длительность", f"{call_data.get('CALL_DURATION', 0)} сек")

            st.markdown("---")

            # Читаем аудиофайл
            try:
                with open(audio_file_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()

                # Отображаем аудиоплеер
                st.audio(audio_bytes, format='audio/mp3')

                # Информация о файле
                file_size = Path(audio_file_path).stat().st_size / (1024 * 1024)  # в МБ
                st.info(f"📁 Размер файла: {file_size:.2f} МБ")

                # Если есть транскрипция, показываем ее тоже
                if call_data.get('transcript'):
                    st.markdown("### 📝 Транскрипция")
                    st.text_area(
                        "Текст разговора",
                        call_data['transcript'],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed"
                    )

                # Если есть анализ, показываем основные моменты
                if call_data.get('analysis'):
                    analysis = call_data['analysis']
                    st.markdown("### 🤖 ИИ Анализ")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Тема:** {analysis.get('topic', 'Не определена')}")
                        st.write(f"**Возражение:** {analysis.get('objection_reason', 'Не обнаружено')}")

                    with col2:
                        result = analysis.get('call_result', {})
                        st.write(f"**Результат:** {result.get('result', 'Не определен')}")
                        st.write(f"**Рекомендация:** {analysis.get('objection_recommendation', 'Нет')}")

            except Exception as e:
                st.error(f"Ошибка при загрузке аудиофайла: {str(e)}")
    else:
        st.error("Аудиофайл не найден или не доступен")


def show_transcript_examples(calls_data):
    """Показывает примеры транскрипций БЕЗ ТОНАЛЬНОСТИ"""
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

                objection_reason = analysis.get('objection_reason')
                if objection_reason:
                    st.write(f"**Возражение:** {objection_reason}")
                    recommendation = analysis.get('objection_recommendation')
                    if recommendation:
                        st.write(f"**Рекомендация:** {recommendation}")
                else:
                    st.write("**Возражения:** Не выявлены")

                key_points = analysis.get('key_points', [])
                if key_points:
                    st.write("**Ключевые моменты:**")
                    for i, point in enumerate(key_points, 1):
                        st.write(f"{i}. {point}")
                else:
                    st.write("**Ключевые моменты:** Не выявлены")


if __name__ == "__main__":
    main()