#!/usr/bin/env python3
"""
Основной класс для анализа звонков Bitrix24
"""

from config import *
from ai_analyzer import LocalAIAnalyzer
from bitrix_api import BitrixAPI
from audio_processor import AudioProcessor
from data_manager import DataManager


class BitrixCallAnalyzer:
    """Основной класс для анализа звонков Bitrix24 с локальными моделями"""

    def __init__(self):
        # Структура папок
        self.base_dir = Path("bitrix_analytics")
        self.base_dir.mkdir(exist_ok=True)
        self.reports_dir = self.base_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        # Инициализация компонентов
        self.ai_analyzer = LocalAIAnalyzer()
        self.bitrix_api = BitrixAPI()
        self.audio_processor = AudioProcessor(self.base_dir)
        self.data_manager = DataManager(self.base_dir)

        logger.info("Инициализация BitrixCallAnalyzer завершена")

    def analyze_transcript(self, transcript: str, call_info: Dict) -> Dict:
        """Анализирует транскрипцию звонка с помощью локальных моделей"""
        if not transcript or transcript in ["Ошибка транскрибации", "Не удалось распознать речь"]:
            return {
                "topic": "Неопределенная тема",
                "key_points": [],
                "rejection_reason": None
            }

        try:
            logger.info("Анализ транскрипции с помощью локальных моделей...")

            # Определяем тему (улучшенный алгоритм)
            topic = self.ai_analyzer.classify_topic(transcript)

            # Ищем причину отказа (улучшенный алгоритм)
            rejection_reason = self.ai_analyzer.find_rejection_reason(transcript)

            # Извлекаем ключевые моменты
            key_points = self.ai_analyzer.extract_key_points(transcript)

            result = {
                "topic": topic,
                "key_points": key_points,
                "rejection_reason": rejection_reason
            }

            logger.info(f"Анализ завершен: тема={topic}, отказ={rejection_reason}")
            return result

        except Exception as e:
            logger.error(f"Ошибка анализа транскрипции: {e}")
            return {
                "topic": "Ошибка анализа",
                "key_points": [],
                "rejection_reason": None
            }

    def process_calls_for_date(self, target_date: datetime.datetime) -> List[Dict]:
        """Обрабатывает звонки за дату - полный цикл"""
        date_str = target_date.strftime("%Y-%m-%d")

        # Проверяем кеш
        if self.data_manager.is_data_cached(target_date):
            logger.info(f"Используем кешированные данные за {date_str}")
            return self.data_manager.load_calls_from_cache(target_date)

        # Загружаем новые данные
        if not self.bitrix_api.webhook_url:
            logger.warning("Bitrix24 не настроен, используем тестовые данные")
            calls = self.data_manager.get_test_data(target_date)
        else:
            # Авторизуемся
            if self.bitrix_api.username and self.bitrix_api.password:
                self.bitrix_api.authenticate_bitrix()

            # Получаем звонки
            calls = self.bitrix_api.get_all_calls_for_day(target_date)

            if not calls:
                return []

            # Получаем имена пользователей
            user_ids = {call['PORTAL_USER_ID'] for call in calls if call.get('PORTAL_USER_ID')}
            user_names = self.bitrix_api.get_user_names(user_ids)

            # Добавляем имена к звонкам и определяем направление
            for call in calls:
                user_id = call.get('PORTAL_USER_ID', '')
                call['user_name'] = user_names.get(user_id, 'Unknown')
                call['call_direction'] = self.bitrix_api.determine_call_direction(call)

            # Извлекаем аудиозаписи если авторизованы
            if self.bitrix_api.authenticated:
                calls = self._extract_audio_recordings(calls, user_names, date_str)

        # Транскрибация и анализ
        for call in calls:
            if 'audio_file' in call:
                # Транскрибация
                filename = Path(call['audio_file']).stem
                existing_transcript = self.audio_processor.load_transcript(filename, date_str)

                if existing_transcript:
                    call['transcript'] = existing_transcript
                else:
                    logger.info(f"Транскрибируем: {call.get('audio_filename', 'аудио')}")
                    transcript = self.audio_processor.transcribe_audio(call['audio_file'])
                    call['transcript'] = transcript
                    self.audio_processor.save_transcript(transcript, filename, date_str)

                # Анализ
                existing_analysis = self.data_manager.load_analysis(filename, date_str)

                if existing_analysis:
                    call['analysis'] = existing_analysis
                else:
                    logger.info(f"Анализируем: {call.get('audio_filename', 'транскрипцию')}")
                    analysis = self.analyze_transcript(call['transcript'], call)
                    call['analysis'] = analysis
                    self.data_manager.save_analysis(analysis, filename, date_str)

            # Для тестовых данных добавляем анализ
            elif call.get('transcript') and not call.get('analysis'):
                call['analysis'] = self.analyze_transcript(call['transcript'], call)

        # Сохраняем в кеш
        self.data_manager.save_calls_to_cache(target_date, calls)

        return calls

    def _extract_audio_recordings(self, calls: List[Dict], user_names: Dict[str, str], date_str: str) -> List[Dict]:
        """Извлекает аудиозаписи звонков с улучшенным определением направления"""
        logger.info("Извлечение аудиозаписей...")

        activity_ids = [call.get('CRM_ACTIVITY_ID') for call in calls if call.get('CRM_ACTIVITY_ID')]

        if not activity_ids:
            return calls

        calls_map = {call.get('CRM_ACTIVITY_ID'): call for call in calls if call.get('CRM_ACTIVITY_ID')}

        # Поиск активностей с файлами
        for aid in activity_ids:
            activity_data = self.bitrix_api.make_api_call("crm.activity.get", {"id": aid})
            if activity_data and 'result' in activity_data:
                activity = activity_data['result']
                call = calls_map.get(aid)
                if call and 'FILES' in activity and activity['FILES']:
                    # Формируем имя файла
                    call_time = call.get('CALL_START_DATE', '')
                    if call_time:
                        try:
                            utc_time = datetime.datetime.fromisoformat(call_time.replace('Z', '+00:00'))
                            local_time = utc_time.astimezone(pytz.timezone(PORTAL_TIMEZONE))
                            time_str = local_time.strftime('%H-%M-%S')
                        except:
                            time_str = "unknown-time"
                    else:
                        time_str = "unknown-time"

                    user_id = call.get('PORTAL_USER_ID', '')
                    user_name = user_names.get(user_id, f'user_{user_id}').replace(' ', '_')

                    phone_number = call.get('PHONE_NUMBER', 'unknown_number')
                    clean_phone = ''.join(filter(str.isdigit, phone_number))

                    direction = call.get('call_direction', 'unknown')
                    filename = f"{date_str}_{time_str}_{direction}_{user_name}_{clean_phone}_id{aid}"

                    for file_info in activity['FILES']:
                        file_path = self.audio_processor.download_audio_file(
                            file_info, filename, date_str, self.bitrix_api.session
                        )
                        if file_path:
                            call['audio_file'] = file_path
                            call['audio_filename'] = f"{filename}.mp3"
                            break

        return calls

    def generate_pdf_report(self, all_calls: List[Dict], date_range: str) -> Optional[bytes]:
        """Генерирует PDF отчет"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            import io

            # Создаем PDF в памяти
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)

            # Стили
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
                alignment=1
            )

            story = []

            # Заголовок
            story.append(Paragraph("Отчет по звонкам Bitrix24", title_style))
            story.append(Paragraph(f"Период: {date_range}", styles['Normal']))
            story.append(Paragraph(f"Создан: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}", styles['Normal']))
            story.append(Spacer(1, 20))

            # Общая статистика
            total_calls = len(all_calls)
            incoming_calls = sum(1 for call in all_calls if call.get('call_direction') == 'incoming')
            outgoing_calls = sum(1 for call in all_calls if call.get('call_direction') == 'outgoing')

            stats_data = [
                ['Показатель', 'Значение'],
                ['Всего звонков', str(total_calls)],
                ['Входящие звонки', str(incoming_calls)],
                ['Исходящие звонки', str(outgoing_calls)],
                ['С аудиозаписью', str(sum(1 for call in all_calls if 'audio_file' in call))]
            ]

            stats_table = Table(stats_data)
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(Paragraph("Общая статистика", styles['Heading2']))
            story.append(stats_table)
            story.append(Spacer(1, 20))

            # Темы звонков
            topics = {}
            for call in all_calls:
                analysis = call.get('analysis', {})
                topic = analysis.get('topic', 'Неопределенная тема')
                topics[topic] = topics.get(topic, 0) + 1

            if topics:
                story.append(Paragraph("Темы звонков", styles['Heading2']))
                topic_data = [['Тема', 'Количество']]
                for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True):
                    topic_data.append([topic, str(count)])

                topic_table = Table(topic_data)
                topic_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))

                story.append(topic_table)

            # Создаем PDF
            doc.build(story)

            # Возвращаем байты PDF
            pdf_bytes = buffer.getvalue()
            buffer.close()

            return pdf_bytes

        except ImportError:
            logger.error("Для экспорта PDF установите: pip install reportlab")
            return None
        except Exception as e:
            logger.error(f"Ошибка создания PDF: {e}")
            return None