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
        """Генерирует тестовые данные для демонстрации"""
        test_transcripts = [
            "Здравствуйте! Меня интересует ваш продукт. Какая цена? Понятно, но это слишком дорого для нас. Спасибо, пока.",
            "Добрый день! У нас проблема с системой. Не работает модуль отчетов. Нужна срочная помощь технической поддержки.",
            "Привет! Хочу заказать вашу услугу. Когда можете начать? Отлично, договорились!",
            "Звоню по поводу партнерства. Наша компания готова к сотрудничеству. Давайте встретимся.",
            "У нас жалоба на качество обслуживания. Очень недовольны. Требуем возврат денег!"
        ]

        test_calls = []
        for i in range(5):
            call_time = target_date + datetime.timedelta(hours=9 + i, minutes=i * 10)

            test_call = {
                'ID': f'test_{i + 1}',
                'CRM_ACTIVITY_ID': f'activity_{i + 1}',
                'CALL_START_DATE': call_time.isoformat(),
                'CALL_TYPE': 1 if i % 2 == 0 else 2,
                'PHONE_NUMBER': f'+7900123456{i}',
                'PORTAL_USER_ID': '1',
                'user_name': 'Тестовый Менеджер',
                'CALL_DURATION': 120 + i * 30,
                'test_data': True,
                'transcript': test_transcripts[i],
                'call_direction': 'incoming' if i % 2 == 0 else 'outgoing'
            }

            test_calls.append(test_call)

        logger.info(f"Сгенерировано {len(test_calls)} тестовых звонков")
        return test_calls