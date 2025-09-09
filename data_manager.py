#!/usr/bin/env python3
"""
Класс для управления данными и кешированием
"""

from config import *


class DataManager:
    """Класс для управления данными и кешированием"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.cache_dir = base_dir / "cache"
        self.analysis_dir = base_dir / "analysis"

        # Создаем папки
        for dir_path in [self.cache_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_date_hash(self, target_date: datetime.datetime) -> str:
        """Создает хеш для даты для кеширования"""
        date_str = target_date.strftime("%Y-%m-%d")
        return hashlib.md5(date_str.encode()).hexdigest()[:8]

    def is_data_cached(self, target_date: datetime.datetime) -> bool:
        """Проверяет, есть ли кешированные данные за дату"""
        date_str = target_date.strftime("%Y-%m-%d")
        cache_file = self.cache_dir / f"calls_{date_str}.json"
        return cache_file.exists()

    def save_calls_to_cache(self, target_date: datetime.datetime, calls: List[Dict]):
        """Сохраняет звонки в кеш"""
        date_str = target_date.strftime("%Y-%m-%d")
        cache_file = self.cache_dir / f"calls_{date_str}.json"

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(calls, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Данные сохранены в кеш: {cache_file}")

    def load_calls_from_cache(self, target_date: datetime.datetime) -> List[Dict]:
        """Загружает звонки из кеша"""
        date_str = target_date.strftime("%Y-%m-%d")
        cache_file = self.cache_dir / f"calls_{date_str}.json"

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                calls = json.load(f)
            logger.info(f"Данные загружены из кеша: {cache_file}")
            return calls
        except Exception as e:
            logger.error(f"Ошибка загрузки из кеша: {e}")
            return []

    def save_analysis(self, analysis: Dict, filename: str, date_str: str):
        """Сохраняет результат анализа"""
        analysis_date_dir = self.analysis_dir / date_str
        analysis_date_dir.mkdir(exist_ok=True)

        analysis_file = analysis_date_dir / f"{filename}_analysis.json"

        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

    def load_analysis(self, filename: str, date_str: str) -> Optional[Dict]:
        """Загружает результат анализа"""
        analysis_file = self.analysis_dir / date_str / f"{filename}_analysis.json"

        if analysis_file.exists():
            try:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки анализа: {e}")

        return None

    def get_test_data(self, target_date: datetime.datetime) -> List[Dict]:
        """Генерирует тестовые данные для демонстрации с примерами различных возражений"""
        test_transcripts = [
            # Пример с возражением по цене
            "Здравствуйте! Меня интересует ваш продукт. Какая цена? 150 тысяч? Понятно, но это слишком дорого для нас. У нас бюджет гораздо меньше. Спасибо, пока.",

            # Пример с возражением по времени
            "Добрый день! У нас проблема с системой. Не работает модуль отчетов. Нужна срочная помощь технической поддержки. Но сейчас времени нет разговаривать, очень занят. Перезвоните позже.",

            # Пример успешного разговора без возражений
            "Привет! Хочу заказать вашу услугу. Когда можете начать? Отлично, договорились! Жду коммерческое предложение на почту.",

            # Пример с возражением по недоверию
            "Звоню по поводу партнерства. Но я первый раз слышу о вашей компании. Не уверен в надежности. Нет гарантий. Сомневаюсь, что стоит рисковать.",

            # Пример с возражением - уже есть поставщик
            "У нас жалоба на качество обслуживания. Кстати, мы уже работаем с другой компанией по этому направлению. У нас постоянный поставщик. Договор уже заключен.",

            # Пример с возражением - нужно подумать
            "Интересное предложение, но мне нужно посоветоваться с руководством. Такое решение я сам принять не могу. Рассмотрим ваше предложение и свяжемся.",

            # Пример с возражением - условия не подходят
            "Ваши условия нас не устраивают. Сроки не подходят, требования другие. Нужны совершенно иные условия сотрудничества.",

            # Пример с возражением по функциональности
            "Продукт неплохой, но не хватает функций. Не все возможности есть. Нужны дополнительные опции, которых у вас пока нет.",

            # Пример с возражением по компетенциям
            "Сомневаюсь в экспертизе вашей команды. Мало опыта в нашей отрасли. Нужны специалисты с большей квалификацией.",

            # Пример с возражением - сравнение с конкурентами
            "У конкурентов условия лучше. Другие компании предлагают дешевле. Есть более выгодные варианты на рынке."
        ]

        test_calls = []
        for i in range(len(test_transcripts)):
            call_time = target_date + datetime.timedelta(hours=9 + i, minutes=i * 15)

            test_call = {
                'ID': f'test_{i + 1}',
                'CRM_ACTIVITY_ID': f'activity_{i + 1}',
                'CALL_START_DATE': call_time.isoformat(),
                'CALL_TYPE': 1 if i % 2 == 0 else 2,
                'PHONE_NUMBER': f'+7900123456{i}',
                'PORTAL_USER_ID': '1',
                'user_name': f'Менеджер {(i % 3) + 1}',  # Три разных менеджера
                'CALL_DURATION': 120 + i * 30,
                'test_data': True,
                'transcript': test_transcripts[i],
                'call_direction': 'incoming' if i % 2 == 0 else 'outgoing'
            }

            test_calls.append(test_call)

        logger.info(f"Сгенерировано {len(test_calls)} тестовых звонков с примерами возражений")
        return test_calls

    def export_custom_objections(self, filepath: str = None) -> str:
        """Экспортирует пользовательские категории возражений в JSON"""
        if not filepath:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.base_dir / f"custom_objections_export_{timestamp}.json"

        custom_objections_file = self.base_dir / "custom_objections.json"

        try:
            if custom_objections_file.exists():
                with open(custom_objections_file, 'r', encoding='utf-8') as f:
                    custom_objections = json.load(f)
            else:
                custom_objections = {}

            export_data = {
                "exported_at": datetime.datetime.now().isoformat(),
                "total_custom_categories": len(custom_objections),
                "categories": custom_objections
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Пользовательские категории возражений экспортированы в: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Ошибка экспорта пользовательских возражений: {e}")
            return ""

    def import_custom_objections(self, filepath: str) -> bool:
        """Импортирует пользовательские категории возражений из JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            if 'categories' not in import_data:
                logger.error("Неверный формат файла импорта")
                return False

            custom_objections_file = self.base_dir / "custom_objections.json"

            # Загружаем существующие категории
            existing_objections = {}
            if custom_objections_file.exists():
                with open(custom_objections_file, 'r', encoding='utf-8') as f:
                    existing_objections = json.load(f)

            # Объединяем с импортированными
            imported_categories = import_data['categories']
            merged_objections = {**existing_objections, **imported_categories}

            # Сохраняем объединенные категории
            with open(custom_objections_file, 'w', encoding='utf-8') as f:
                json.dump(merged_objections, f, ensure_ascii=False, indent=2)

            imported_count = len(imported_categories)
            total_count = len(merged_objections)

            logger.info(f"Импортировано {imported_count} категорий. Всего категорий: {total_count}")
            return True

        except Exception as e:
            logger.error(f"Ошибка импорта пользовательских возражений: {e}")
            return False

    def cleanup_old_cache(self, days_to_keep: int = 30):
        """Очищает старые кеш-файлы старше указанного количества дней"""
        try:
            current_time = datetime.datetime.now()
            cutoff_time = current_time - datetime.timedelta(days=days_to_keep)

            deleted_count = 0
            for cache_file in self.cache_dir.glob("calls_*.json"):
                if cache_file.stat().st_mtime < cutoff_time.timestamp():
                    cache_file.unlink()
                    deleted_count += 1

            logger.info(f"Удалено {deleted_count} старых кеш-файлов")

        except Exception as e:
            logger.error(f"Ошибка очистки кеша: {e}")

    def get_cache_statistics(self) -> Dict:
        """Возвращает статистику по кешу"""
        try:
            cache_files = list(self.cache_dir.glob("calls_*.json"))
            total_files = len(cache_files)

            if total_files == 0:
                return {"total_files": 0, "total_size_mb": 0, "date_range": None}

            # Подсчитываем общий размер
            total_size = sum(f.stat().st_size for f in cache_files)
            total_size_mb = total_size / (1024 * 1024)

            # Находим диапазон дат
            dates = []
            for cache_file in cache_files:
                try:
                    date_str = cache_file.stem.replace("calls_", "")
                    dates.append(datetime.datetime.strptime(date_str, "%Y-%m-%d"))
                except:
                    continue

            date_range = None
            if dates:
                dates.sort()
                date_range = {
                    "from": dates[0].strftime("%Y-%m-%d"),
                    "to": dates[-1].strftime("%Y-%m-%d")
                }

            return {
                "total_files": total_files,
                "total_size_mb": round(total_size_mb, 2),
                "date_range": date_range
            }

        except Exception as e:
            logger.error(f"Ошибка получения статистики кеша: {e}")
            return {"total_files": 0, "total_size_mb": 0, "date_range": None}