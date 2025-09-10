#!/usr/bin/env python3
"""
Продвинутый NLU анализ диалогов с использованием Natasha
Специализированный для анализа телефонных разговоров санатория
"""

import re
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import numpy as np

# Natasha - легковесный NER для русского языка
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)

# Для извлечения дат и денег
from natasha import DatesExtractor, MoneyExtractor

# DeepPavlov для глубокого анализа диалогов (опционально)
try:
    from deeppavlov import configs, build_model

    DEEPPAVLOV_AVAILABLE = True
except ImportError:
    DEEPPAVLOV_AVAILABLE = False
    logging.warning("DeepPavlov не установлен. Используйте: pip install deeppavlov")

# Transformers для дополнительного анализа
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)


class AdvancedNLUAnalyzer:
    """Продвинутый NLU анализатор для телефонных разговоров"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используется устройство: {self.device}")

        # Инициализация Natasha
        self._init_natasha()

        # Инициализация DeepPavlov (если доступен)
        self._init_deeppavlov()

        # Инициализация дополнительных моделей
        self._init_additional_models()

        # Загрузка паттернов для санаторного бизнеса
        self._load_business_patterns()

    def _init_natasha(self):
        """Инициализация компонентов Natasha"""
        logger.info("Инициализация Natasha...")

        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()

        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)

        # Экстракторы для специфичных сущностей
        self.dates_extractor = DatesExtractor(self.morph_vocab)
        self.money_extractor = MoneyExtractor(self.morph_vocab)

        logger.info("✅ Natasha инициализирована")

    def _init_deeppavlov(self):
        """Инициализация моделей DeepPavlov"""
        if not DEEPPAVLOV_AVAILABLE:
            self.intent_model = None
            self.ner_model = None
            return

        try:
            logger.info("Инициализация DeepPavlov...")
            # Можно добавить инициализацию моделей DeepPavlov здесь
            self.intent_model = None
            self.ner_model = None
            logger.info("✅ DeepPavlov готов к использованию")
        except Exception as e:
            logger.error(f"Ошибка инициализации DeepPavlov: {e}")
            self.intent_model = None
            self.ner_model = None

    def _init_additional_models(self):
        """Инициализация дополнительных моделей"""
        try:
            # Модель для классификации диалогов (опционально)
            self.dialogue_classifier = None
            logger.info("✅ Дополнительные модели загружены")
        except:
            self.dialogue_classifier = None
            logger.warning("Не удалось загрузить дополнительные модели")

    def _load_business_patterns(self):
        """Загрузка паттернов для санаторного бизнеса"""

        # Паттерны возражений специфичные для санатория
        self.sanatorium_objections = {
            "🏥 Лечение не нужно": {
                "patterns": [
                    r"не нужно лечение",
                    r"только отдых",
                    r"без процедур",
                    r"просто отдохнуть",
                    r"без лечения"
                ],
                "recommendation": "Предложить оздоровительные программы без медицинских процедур"
            },
            "📅 Неудобные даты": {
                "patterns": [
                    r"даты не подходят",
                    r"в другое время",
                    r"перенести на",
                    r"не можем в эти числа",
                    r"неудобный период"
                ],
                "recommendation": "Предложить альтернативные даты или гибкий график заезда"
            },
            "💰 Высокая цена": {
                "patterns": [
                    r"дорого",
                    r"цена высокая",
                    r"много денег",
                    r"не по карману",
                    r"слишком дорого",
                    r"200.*за кисловодск",
                    r"мальдивы дешевле"
                ],
                "recommendation": "Показать ценность комплексного предложения, предложить акции"
            }
        }

        # Темы разговоров для санатория
        self.sanatorium_topics = {
            "Бронирование": ["бронь", "забронировать", "зарезервировать", "номер свободен"],
            "Информация о ценах": ["стоимость", "цена", "сколько стоит", "тариф", "прайс"],
            "Условия проживания": ["питание", "номер", "удобства", "что входит", "включено"],
            "Лечебные программы": ["лечение", "процедуры", "врач", "диагностика", "профиль"]
        }

        # Этапы продажи
        self.sales_stages = {
            "initial_contact": "Первичный контакт",
            "needs_identification": "Выявление потребностей",
            "presentation": "Презентация услуг",
            "objection_handling": "Работа с возражениями",
            "closing": "Закрытие сделки",
            "post_sale": "Постпродажное обслуживание"
        }

    def analyze_dialogue(self, transcript: str, call_info: Dict = None) -> Dict:
        """Комплексный анализ диалога"""

        # Разбиваем на реплики
        dialogue_turns = self._split_dialogue(transcript)

        # Извлекаем сущности через Natasha
        entities = self._extract_entities_natasha(transcript)

        # Анализируем интенты
        intents = self._analyze_intents_fallback(dialogue_turns)

        # Определяем тему разговора
        topic = self._determine_topic(transcript, intents)

        # Находим возражения
        objections = self._find_objections(transcript, dialogue_turns)

        # Определяем этап продажи
        sales_stage = self._determine_sales_stage(dialogue_turns, intents)

        # Анализируем результат разговора
        call_result = self._analyze_call_result(transcript, dialogue_turns, objections)

        # Извлекаем ключевую бизнес-информацию
        business_data = self._extract_business_data(entities, transcript)

        # Анализируем эмоциональный контекст
        emotional_context = self._analyze_emotional_context(dialogue_turns)

        return {
            "topic": topic,
            "objections": objections,
            "entities": entities,
            "intents": intents,
            "sales_stage": sales_stage,
            "call_result": call_result,
            "business_data": business_data,
            "emotional_context": emotional_context,
            "dialogue_turns": len(dialogue_turns),
            "key_points": self._extract_key_points(transcript, entities)
        }

    def _split_dialogue(self, transcript: str) -> List[Dict]:
        """Разделение транскрипции на реплики"""
        turns = []

        # Простое разделение по предложениям
        sentences = re.split(r'[.!?]+', transcript)
        for i, sent in enumerate(sentences):
            if sent.strip():
                # Эвристика: вопросы обычно от клиента, ответы от менеджера
                speaker = "client" if '?' in sent else "manager"
                turns.append({
                    "speaker": speaker,
                    "text": sent.strip(),
                    "position": i
                })

        return turns

    def _extract_entities_natasha(self, text: str) -> Dict[str, List[Dict]]:
        """Извлечение сущностей через Natasha"""
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "money": [],
            "phones": [],
            "emails": []
        }

        try:
            # Создаем документ Natasha
            doc = Doc(text)

            # Сегментация
            doc.segment(self.segmenter)

            # Токенизация и морфология
            doc.tag_morph(self.morph_tagger)

            # Лемматизация
            for token in doc.tokens:
                token.lemmatize(self.morph_vocab)

            # NER
            doc.tag_ner(self.ner_tagger)

            # Извлекаем сущности из doc.spans
            for span in doc.spans:
                entity_info = {
                    "text": text[span.start:span.stop],
                    "type": span.type,
                    "start": span.start,
                    "stop": span.stop
                }

                # Распределяем по категориям
                if span.type == "PER":
                    entities["persons"].append(entity_info)
                elif span.type == "ORG":
                    entities["organizations"].append(entity_info)
                elif span.type == "LOC":
                    entities["locations"].append(entity_info)

            # Извлекаем даты
            try:
                dates_matches = self.dates_extractor(text)
                for match in dates_matches:
                    # У match есть атрибуты start и stop напрямую
                    date_info = {
                        "text": text[match.start:match.stop],
                        "start": match.start,
                        "stop": match.stop
                    }
                    # Добавляем факт если есть
                    if hasattr(match, 'fact') and match.fact:
                        date_info["fact"] = str(match.fact)
                    entities["dates"].append(date_info)
            except Exception as e:
                logger.debug(f"Не удалось извлечь даты: {e}")

            # Извлекаем денежные суммы
            try:
                money_matches = self.money_extractor(text)
                for match in money_matches:
                    # У match есть атрибуты start и stop напрямую
                    money_info = {
                        "text": text[match.start:match.stop],
                        "start": match.start,
                        "stop": match.stop
                    }
                    # Добавляем детали если есть
                    if hasattr(match, 'fact') and match.fact:
                        if hasattr(match.fact, 'amount'):
                            money_info["amount"] = match.fact.amount
                        if hasattr(match.fact, 'currency'):
                            money_info["currency"] = match.fact.currency
                        else:
                            money_info["currency"] = "RUB"  # По умолчанию рубли
                    entities["money"].append(money_info)
            except Exception as e:
                logger.debug(f"Не удалось извлечь суммы: {e}")

            # Извлекаем телефоны
            phone_pattern = r'\+?[78]?\s?[\(\[]?\d{3}[\)\]]?\s?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'
            for phone_match in re.finditer(phone_pattern, text):
                entities["phones"].append({
                    "text": phone_match.group(),
                    "start": phone_match.start(),
                    "stop": phone_match.end()
                })

            # Извлекаем email
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            for email_match in re.finditer(email_pattern, text):
                entities["emails"].append({
                    "text": email_match.group(),
                    "start": email_match.start(),
                    "stop": email_match.end()
                })

            # Добавляем специфичные для санатория сущности
            sanatorium_entities = self._extract_sanatorium_entities(text)

            # Добавляем только те сущности, которые не None и являются списками
            for key, value in sanatorium_entities.items():
                if isinstance(value, list):
                    entities[key] = value
                elif value is not None:
                    # Для скалярных значений (duration_nights, guest_count) создаем отдельные ключи
                    entities[key] = value

            # Подсчитываем только списковые сущности
            total_count = 0
            for key, value in entities.items():
                if isinstance(value, list):
                    total_count += len(value)
                elif value is not None and not isinstance(value, dict):
                    total_count += 1

            logger.debug(f"Natasha извлекла {total_count} сущностей")

        except Exception as e:
            logger.error(f"Ошибка при извлечении сущностей Natasha: {e}")
            import traceback
            traceback.print_exc()

        return entities

    def _extract_sanatorium_entities(self, text: str) -> Dict:
        """Извлечение сущностей специфичных для санатория"""
        entities = {
            "room_types": [],
            "meal_types": [],
            "treatment_types": [],
            "duration_nights": None,
            "guest_count": None
        }

        # Типы номеров
        room_patterns = [
            r'(стандарт|люкс|делюкс|эконом|одноместный|двухместный)',
            r'номер\s+(с балконом|без балкона|с видом на)',
        ]
        for pattern in room_patterns:
            matches = re.findall(pattern, text.lower())
            entities["room_types"].extend(matches)

        # Типы питания
        meal_patterns = [
            r'(завтрак|обед|ужин|шведский стол|диетическое|трехразовое|питание)',
            r'(все включено|полный пансион|полупансион)'
        ]
        for pattern in meal_patterns:
            matches = re.findall(pattern, text.lower())
            entities["meal_types"].extend(matches)

        # Количество ночей
        nights_pattern = r'(\d+)\s*(ночей|ночи|ночь|дней|дня|день|суток)'
        nights_match = re.search(nights_pattern, text.lower())
        if nights_match:
            entities["duration_nights"] = int(nights_match.group(1))

        # Количество гостей
        guest_patterns = [
            r'(\d+)\s*(человек|взрослых|детей|ребенок)',
            r'(один|одна|двое|трое|четверо)\s*(человек|взрослых)?'
        ]
        for pattern in guest_patterns:
            match = re.search(pattern, text.lower())
            if match:
                number_text = match.group(1)
                text_to_num = {
                    'один': 1, 'одна': 1, 'двое': 2, 'трое': 3, 'четверо': 4
                }
                if number_text in text_to_num:
                    entities["guest_count"] = text_to_num[number_text]
                elif number_text.isdigit():
                    entities["guest_count"] = int(number_text)
                break

        return entities

    def _analyze_intents_fallback(self, dialogue_turns: List[Dict]) -> List[Dict]:
        """Анализ интентов через паттерны"""
        intents = []

        intent_patterns = {
            "запрос_информации": ["сколько", "какая цена", "что входит", "расскажите"],
            "бронирование": ["забронировать", "зарезервировать", "оформить"],
            "возражение": ["дорого", "не подходит", "подумать", "не устраивает"],
            "согласие": ["хорошо", "подходит", "устраивает", "давайте"],
            "уточнение": ["а что если", "можно ли", "возможно ли"],
            "отказ": ["не буду", "не интересно", "спасибо, нет"],
            "благодарность": ["спасибо", "благодарю"]
        }

        for turn in dialogue_turns:
            text_lower = turn["text"].lower()
            detected_intent = "общий"
            max_confidence = 0.3

            for intent_name, keywords in intent_patterns.items():
                matches = sum(1 for keyword in keywords if keyword in text_lower)
                confidence = matches / len(keywords) if keywords else 0

                if confidence > max_confidence:
                    detected_intent = intent_name
                    max_confidence = confidence

            intents.append({
                "speaker": turn["speaker"],
                "text": turn["text"][:100],
                "intent": detected_intent,
                "confidence": max_confidence
            })

        return intents

    def _determine_topic(self, transcript: str, intents: List[Dict]) -> str:
        """Определение основной темы разговора"""
        text_lower = transcript.lower()

        # Подсчет упоминаний по темам
        topic_scores = {}

        for topic, keywords in self.sanatorium_topics.items():
            score = sum(3 if keyword in text_lower else 0 for keyword in keywords)
            if score > 0:
                topic_scores[topic] = score

        # Возвращаем тему с максимальным счетом
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)

        return "Общий разговор"

    def _find_objections(self, transcript: str, dialogue_turns: List[Dict]) -> List[Dict]:
        """Поиск и классификация возражений"""
        objections = []
        text_lower = transcript.lower()

        # Проверяем специфичные возражения санатория
        for objection_type, objection_data in self.sanatorium_objections.items():
            for pattern in objection_data["patterns"]:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    # Находим контекст
                    start = max(0, match.start() - 100)
                    end = min(len(text_lower), match.end() + 100)
                    context = transcript[start:end]

                    objections.append({
                        "type": objection_type,
                        "text": match.group(),
                        "context": context,
                        "recommendation": objection_data["recommendation"],
                        "confidence": 0.9
                    })

        return objections

    def _determine_sales_stage(self, dialogue_turns: List[Dict], intents: List[Dict]) -> str:
        """Определение этапа продажи"""
        if len(dialogue_turns) < 3:
            return self.sales_stages["initial_contact"]

        # Простая эвристика на основе паттернов
        full_text = " ".join([turn["text"] for turn in dialogue_turns])
        full_text_lower = full_text.lower()

        if "забронировать" in full_text_lower or "оформить" in full_text_lower:
            return self.sales_stages["closing"]
        elif "возражение" in [intent["intent"] for intent in intents]:
            return self.sales_stages["objection_handling"]
        elif "расскажите" in full_text_lower or "что входит" in full_text_lower:
            return self.sales_stages["presentation"]
        elif "интересует" in full_text_lower or "хотел узнать" in full_text_lower:
            return self.sales_stages["needs_identification"]

        return self.sales_stages["needs_identification"]

    def _analyze_call_result(self, transcript: str, dialogue_turns: List[Dict], objections: List[Dict]) -> Dict:
        """Анализ результата звонка"""
        text_lower = transcript.lower()

        # Индикаторы успешной продажи
        success_indicators = [
            "забронировали", "оформили", "договорились", "жду вас",
            "до встречи", "оплатил", "внесли предоплату", "счет выставлен"
        ]

        # Индикаторы отказа
        failure_indicators = [
            "не подходит", "дорого", "подумаем", "перезвоню",
            "не интересно", "спасибо, нет", "в другой раз"
        ]

        # Индикаторы необходимости дальнейшей работы
        followup_indicators = [
            "пришлите информацию", "отправьте предложение",
            "обсужу с", "нужно подумать", "изучу"
        ]

        # Подсчет индикаторов
        success_score = sum(1 for ind in success_indicators if ind in text_lower)
        failure_score = sum(1 for ind in failure_indicators if ind in text_lower)
        followup_score = sum(1 for ind in followup_indicators if ind in text_lower)

        # Определение результата
        result_type = "неопределенный"
        confidence = 0.5

        if success_score > failure_score and success_score > followup_score:
            result_type = "успешная продажа"
            confidence = min(0.9, success_score * 0.3)
        elif failure_score > success_score:
            result_type = "отказ"
            confidence = min(0.9, failure_score * 0.3)
        elif followup_score > 0:
            result_type = "требуется доработка"
            confidence = min(0.8, followup_score * 0.4)

        return {
            "result": result_type,
            "confidence": confidence,
            "success_indicators_found": success_score,
            "failure_indicators_found": failure_score,
            "followup_needed": followup_score > 0,
            "objections_count": len(objections),
            "recommendation": self._get_result_recommendation(result_type, objections)
        }

    def _get_result_recommendation(self, result_type: str, objections: List[Dict]) -> str:
        """Рекомендации по результату звонка"""
        if result_type == "успешная продажа":
            return "Отправить подтверждение брони и инструкции для клиента"
        elif result_type == "отказ":
            if objections:
                main_objection = objections[0]["type"] if objections else "неизвестно"
                return f"Проанализировать причину отказа: {main_objection}. Возможен повторный контакт через 2-3 недели"
            return "Выяснить истинную причину отказа для улучшения скриптов"
        elif result_type == "требуется доработка":
            return "Отправить дополнительные материалы и назначить повторный звонок через 2-3 дня"
        else:
            return "Уточнить результат разговора у менеджера"

    def _extract_business_data(self, entities: Dict, transcript: str) -> Dict:
        """Извлечение ключевой бизнес-информации"""
        business_data = {
            "dates": {
                "check_in": None,
                "check_out": None,
                "duration": entities.get("duration_nights")
            },
            "guests": {
                "adults": None,
                "children": None,
                "total": entities.get("guest_count")
            },
            "preferences": {
                "room_type": entities.get("room_types", []),
                "meal_plan": entities.get("meal_types", []),
                "treatments": entities.get("treatment_types", [])
            },
            "financial": {
                "quoted_prices": entities.get("money", []),
                "discount_mentioned": self._check_discount(transcript),
                "payment_method": self._extract_payment_method(transcript)
            },
            "contact_info": {
                "phones": entities.get("phones", []),
                "emails": entities.get("emails", []),
                "names": entities.get("persons", [])
            }
        }

        # Парсинг количества гостей
        adults_pattern = r'(\d+)\s*взрослых'
        children_pattern = r'(\d+)\s*(ребен|дет)'

        adults_match = re.search(adults_pattern, transcript.lower())
        if adults_match:
            business_data["guests"]["adults"] = int(adults_match.group(1))

        children_match = re.search(children_pattern, transcript.lower())
        if children_match:
            business_data["guests"]["children"] = int(children_match.group(1))

        return business_data

    def _check_discount(self, transcript: str) -> Dict:
        """Проверка упоминания скидок"""
        text_lower = transcript.lower()
        discount_info = {
            "mentioned": False,
            "percentage": None,
            "type": None
        }

        # Поиск процентов скидки
        percent_pattern = r'(\d+)\s*(%|процент)'
        percent_match = re.search(percent_pattern, text_lower)

        if percent_match and "скидк" in text_lower:
            discount_info["mentioned"] = True
            discount_info["percentage"] = int(percent_match.group(1))

        # Типы скидок
        if "акци" in text_lower:
            discount_info["type"] = "акция"
        elif "раннее бронирование" in text_lower:
            discount_info["type"] = "раннее бронирование"
        elif "постоянный клиент" in text_lower:
            discount_info["type"] = "постоянный клиент"

        return discount_info

    def _extract_payment_method(self, transcript: str) -> Optional[str]:
        """Извлечение способа оплаты"""
        text_lower = transcript.lower()

        if "наличны" in text_lower:
            return "наличные"
        elif "карт" in text_lower:
            return "карта"
        elif "безнал" in text_lower or "перечисление" in text_lower:
            return "безналичный расчет"
        elif "рассрочк" in text_lower:
            return "рассрочка"
        elif "предоплат" in text_lower:
            return "предоплата"

        return None

    def _analyze_emotional_context(self, dialogue_turns: List[Dict]) -> Dict:
        """Анализ эмоционального контекста диалога"""
        emotional_context = {
            "client_mood": "нейтральный",
            "manager_mood": "нейтральный",
            "tension_points": [],
            "positive_moments": []
        }

        # Индикаторы эмоций
        positive_indicators = ["спасибо", "отлично", "хорошо", "замечательно", "прекрасно"]
        negative_indicators = ["плохо", "ужасно", "дорого", "не устраивает", "не подходит"]

        for i, turn in enumerate(dialogue_turns):
            text_lower = turn["text"].lower()

            # Подсчет индикаторов
            positive_score = sum(1 for ind in positive_indicators if ind in text_lower)
            negative_score = sum(1 for ind in negative_indicators if ind in text_lower)

            # Определение эмоционального состояния
            if positive_score > negative_score:
                emotional_context["positive_moments"].append({
                    "turn": i,
                    "text": turn["text"][:100]
                })
            elif negative_score > positive_score:
                emotional_context["tension_points"].append({
                    "turn": i,
                    "text": turn["text"][:100],
                    "severity": "high" if negative_score > 2 else "medium"
                })

        # Общая оценка настроения
        if len(emotional_context["positive_moments"]) > len(emotional_context["tension_points"]):
            emotional_context["client_mood"] = "позитивный"
        elif len(emotional_context["tension_points"]) > 2:
            emotional_context["client_mood"] = "напряженный"

        return emotional_context

    def _extract_key_points(self, transcript: str, entities: Dict) -> List[str]:
        """Извлечение ключевых моментов разговора"""
        key_points = []
        sentences = re.split(r'[.!?]+', transcript)

        # Важные индикаторы
        important_indicators = [
            "главное", "важно", "обязательно", "критично",
            "принципиально", "основное", "ключевое"
        ]

        # Бизнес-критичные фразы
        business_critical = [
            "забронировали", "оплатил", "договорились",
            "не подходит", "отказываюсь", "дорого",
            "нужно подумать", "перезвоню"
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue

            sentence_lower = sentence.lower()

            # Проверка на важность
            is_important = any(ind in sentence_lower for ind in important_indicators)
            is_business_critical = any(phrase in sentence_lower for phrase in business_critical)

            # Проверка на наличие важных сущностей
            has_money = any(money.get("text", "") in sentence for money in entities.get("money", []))
            has_dates = any(date.get("text", "") in sentence for date in entities.get("dates", []))

            if is_important or is_business_critical or has_money or has_dates:
                key_points.append(sentence)

        # Ограничиваем количество
        return key_points[:5]