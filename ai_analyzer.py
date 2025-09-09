#!/usr/bin/env python3
"""
Класс для локального ИИ анализа текста
"""

from config import *


class LocalAIAnalyzer:
    """Класс для локального анализа текста с использованием трансформеров"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используется устройство: {self.device}")
        self._init_models()

    def _init_models(self):
        """Инициализация локальных моделей"""
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("✅ Модель классификации загружена")
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель классификации: {e}")
            self.classifier = None

    def classify_topic(self, text: str) -> str:
        """Улучшенное определение темы разговора"""
        if not text.strip():
            return "Общий разговор"

        text_lower = text.lower()

        # Проверяем по ключевым словам
        topic_scores = {}
        for topic, keywords in CALL_TOPICS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topic_scores[topic] = score

        # Если нашли совпадения по ключевым словам
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            return best_topic

        # Используем ИИ классификацию
        if self.classifier:
            try:
                text_sample = text[:1000]
                result = self.classifier(text_sample, list(CALL_TOPICS.keys()))

                if result and 'labels' in result and len(result['labels']) > 0:
                    confidence = result['scores'][0]
                    if confidence > 0.2:
                        return result['labels'][0]
            except Exception as e:
                logger.error(f"Ошибка ИИ классификации: {e}")

        # Дополнительная эвристика
        if len(text) < 50:
            return "Короткий разговор"
        elif any(word in text_lower for word in ["здравствуйте", "добрый день", "привет"]):
            return "Общий разговор"
        else:
            return "Неопределенная тема"

    def find_rejection_reason(self, text: str) -> Optional[str]:
        """Улучшенный поиск причины отказа"""
        if not text.strip():
            return None

        text_lower = text.lower()

        # Проверяем наличие отказных фраз
        rejection_indicators = [
            "нет", "не", "отказ", "не интересно", "не нужно", "не подходит",
            "не хочу", "не буду", "откажусь", "спасибо, не надо"
        ]

        has_rejection = any(indicator in text_lower for indicator in rejection_indicators)

        if not has_rejection:
            return None

        # Ищем конкретную причину отказа
        reason_scores = {}
        for reason, patterns in REJECTION_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if pattern in text_lower:
                    # Учитываем контекст
                    pattern_pos = text_lower.find(pattern)
                    context = text_lower[max(0, pattern_pos - 100):pattern_pos + 100]
                    if any(reject_word in context for reject_word in rejection_indicators):
                        score += 2  # Высокий вес если рядом с отказом
                    else:
                        score += 1  # Низкий вес если просто упоминание

            if score > 0:
                reason_scores[reason] = score

        if reason_scores:
            best_reason = max(reason_scores, key=reason_scores.get)
            return best_reason

        return "Общий отказ"

    def extract_key_points(self, text: str) -> List[str]:
        """Извлечение ключевых моментов из текста"""
        if not text.strip():
            return []

        sentences = re.split(r'[.!?]+', text)

        key_sentences = []
        important_words = [
            'важно', 'главное', 'основное', 'нужно', 'требуется', 'проблема',
            'вопрос', 'решение', 'предложение', 'цена', 'стоимость', 'условия',
            'договор', 'заказ', 'услуга', 'продукт', 'клиент', 'компания'
        ]

        for sentence in sentences[:10]:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in important_words):
                    key_sentences.append(sentence)

        return key_sentences[:3]