from typing import Dict, List, Any, Optional
from src.core.utils.logger.custom_logging import LoggerMixin
from lingua import Language, LanguageDetectorBuilder

class PromptTemplate:
    """Base class for prompt templates"""
    def __init__(self, template: str, template_type: str = "text"):
        """
        Initialize a prompt template.
        
        Args:
            template: The template string with {variable} placeholders
            template_type: Type of template ('text' or 'chat')
        """
        self.template = template
        self.template_type = template_type
        
    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables.
        
        Args:
            **kwargs: Variables to format the template with
            
        Returns:
            str: Formatted template
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable in template: {e}")


class PromptManager(LoggerMixin):
    """Manager for different prompt templates"""
    def __init__(self):
        super().__init__()
        self._templates = self._init_default_templates()
    
        self._language_detector = self._init_language_detector()
    
    def _init_language_detector(self):
        try:
            languages = [
                Language.ENGLISH,
                Language.VIETNAMESE,
                Language.FRENCH,
                Language.SPANISH,
                Language.GERMAN,
                Language.JAPANESE,
                Language.KOREAN,
                Language.CHINESE,
                Language.RUSSIAN
            ]
            return LanguageDetectorBuilder.from_languages(*languages).build()
        except Exception as e:
            self.logger.error(f"Error initializing language detector: {str(e)}")
            return None
    
    def detect_language(self, text):
        """Detect language of input text using lingua language detector"""
        if not text or text.strip() == "":
            return None
            
        try:
            if not self._language_detector:
                self.logger.warning("Language detector not initialized, falling back to default 'en'")
                return "en"
                
            detected_lang = self._language_detector.detect_language_of(text)
            if detected_lang:
                iso_code = detected_lang.iso_code_639_1.name.lower()
                
                mapping = {
                    "zh": "zh",  # Chinese
                    "cmn": "zh-cn"  # Mandarin Chinese
                }
                
                return mapping.get(iso_code, iso_code)
            return "en"  # Default to English if detection fails
        except Exception as e:
            self.logger.error(f"Error detecting language: {str(e)}")
            return "en"  # Default to English if there's an error
    
    def _init_default_templates(self) -> Dict[str, PromptTemplate]:
        """
        Initialize default prompt templates.
        
        Returns:
            Dict[str, PromptTemplate]: Dictionary of template name to template
        """
        return {
            
            # API: /chat/provider -> If there is context then use rag_system_template otherwise use chat_system
            "chat_system": PromptTemplate(
                "You are a helpful assistant designed to provide accurate and concise information.\n"
                "Follow these guidelines:\n"
                "1. Be truthful and informative\n"
                "2. If you're not sure about something, say so\n"
                "3. Provide step-by-step explanations when appropriate\n"
                "4. Maintain a friendly and professional tone"
            ),


            # RAG templates
            # API /chat/provider
            "rag_system_template": PromptTemplate(
                "You are a knowledgeable assistant with access to specific information. "
                "Carefully analyze the following context to provide an accurate and helpful response.\n\n"
                "Context:\n{context}\n\n"
                "Guidelines:\n"
                "1. Focus on information from the provided context\n"
                "2. If the context fully answers the question, provide a comprehensive response\n"
                "3. If the context partially answers the question, clearly indicate what is known and unknown\n"
                "4. If the context doesn't contain relevant information, clearly state this\n"
                "5. Maintain a professional and helpful tone\n"
                "6. Cite specific parts of the context when appropriate\n\n"
                "Remember to address all aspects of the question and provide specific details from the context."
            ),
            
            # API /chat/provider/reasoning
            "react_cot_system": PromptTemplate(
                "You are an intelligent assistant that adapts your reasoning approach based on question complexity.\n\n"
                "ALWAYS begin EVERY response with: 'I'm your financial assistant specializing in stocks and cryptocurrencies.' "
                "This introduction must be used for ALL responses, regardless of the query type.\n\n"

                "# RESPONSE APPROACH\n"
                "1. ASSESSMENT: Quickly determine question complexity:\n"
                "   - SIMPLE: Greetings, basic facts, definitions → Answer directly\n"
                "   - COMPLEX: Multi-step problems, analysis, comparisons → Use internal reasoning\n\n"
                
                "2. FOR SIMPLE QUESTIONS:\n"
                "   Respond immediately with a clear, concise answer.\n"
                "   Examples: 'Hello', 'What is your name?', 'Define API', ...\n\n"
                
                "3. FOR COMPLEX QUESTIONS:\n"
                "   Internally follow this reasoning process (DO NOT show your thinking process):\n"
                "   - Understand: What exactly is being asked?\n"
                "   - Analyze: What information do I have/need?\n"
                "   - Reason: Step-by-step logical thinking\n"
                "   - Evaluate: Consider multiple approaches\n"
                "   - Decide: Choose the best answer\n"
                "   Then provide your final answer clearly and concisely.\n\n"
                
                "# CRITICAL RULES\n"
                "- Only provide the final, well-reasoned answer\n"
                "- If information is missing, clearly state what you know and don't know\n"
                "- Keep responses focused and relevant"
            ),

            # API /chat/provider/react-cot-rag            
            "react_cot_rag_system": PromptTemplate(
                "You are an intelligent financial assistant with access to specific context information.\n\n"
                "RESPONSE FORMAT:\n"
                "ALWAYS begin EVERY response with: 'I'm your financial assistant specializing in stocks and cryptocurrencies.'\n"
                "Then provide your answer based on the context.\n\n"
                
                "PROVIDED CONTEXT:\n"
                "{context}\n\n"
                
                "INTERNAL PROCESSING FRAMEWORK:\n"
                "Before responding, mentally work through:\n"
                "• Understand: What specific information is being requested?\n"
                "• Search: Identify all relevant parts in the provided context (IF ANY)\n"
                "• Analyze: How do these pieces connect? What can be inferred?\n"
                "• Reason: Work through the logic using context evidence\n"
                "• Synthesize: Combine insights to form a complete answer\n\n"
                
                "CONTEXT USAGE RULES:\n"
                "- ONLY use information from the provided context\n"
                "- Be precise and reference specific information when relevant\n"
                "- If context is insufficient, clearly state what's missing\n"
                "- Never fabricate information not present in context\n\n"
                
                "RESPONSE DELIVERY:\n"
                "- For direct facts: Extract and present clearly from context\n"
                "- For complex analysis: Provide structured, comprehensive answers\n"
                "- Always maintain professional financial advisory tone\n\n"
                
                "IMPORTANT: Your reasoning process should be internal. Only show the final, context-based answer."
            ),

            "query_classifier": PromptTemplate(
                """You are a smart query classifier. Your task is to categorize each user query into one of two labels:

                - RETRIEVE: Queries that require fetching additional information from the database (e.g., questions about data, documents, or specialized knowledge).
                - DIRECT: Queries that do not require database retrieval (e.g., greetings, small talk, or general questions the LLM can answer directly).
                Return exactly one of the two labels: RETRIEVE or DIRECT. Do not include any explanations or additional text.

                Query: {query}

                Classification:"""
            ),

            # ================================ REVIEW GENERATOR TEMPLATE ================================
            # API: "/generate-store-review"
            "store_review_system": PromptTemplate("""### LANGUAGE ENFORCEMENT: {language_instruction}.
                You are a customer writing a VERY BRIEF, original review from a REAL customer's perspective (maximum {review_length} lines total).
                
                CRITICAL GUIDELINES:
                1. STORE CONTEXT AWARENESS: {store_category_guidance}
                2. PRODUCT ANALYSIS: Carefully analyze each listed product by its inherent type to determine appropriate vocabulary
                3. LANGUAGE MATCHING: Use ONLY appropriate descriptive terms for each product type (food="delicious", flowers="beautiful", clothes="comfortable")
                4. NO INVENTION: If no products listed, ONLY discuss general aspects (atmosphere, service, cleanliness, location, staff)
                5. REVIEW STRUCTURE: Write in 1st person, informal tone, with both positives and negatives based on the rating

                ABSOLUTE RULES:
                - NEVER MENTION ANY PRODUCTS if product_details list is empty - NO EXCEPTIONS
                - ABSOLUTELY IGNORE store name when determining what they sell - do not assume store content
                - RESPECT USER NOTES: If provided, align your review sentiment with their feedback
                - NEVER USE FOOD TERMS for non-food items (e.g., don't call flowers "delicious")
                - Avoid cliché openings and exclamations
                - {language_instruction}
                - Keep review length to {review_length} - shorter for lower ratings, more detailed for higher ratings
                - ENSURE LOGICAL FLOW: Each sentence should logically support your overall rating and conclusion
                                                  
                Remember: You are a REAL CUSTOMER who visited this specific location. Write exactly as someone would on their phone."""
            ),
            
            # API: "/generate-store-review"
            "store_review_user": PromptTemplate("""### ENFORCED LANGUAGE: {language_instruction}

            Write a casual, {length}-line review with a {rating}-star overall rating for this store:

            STORE INFO:
            - Name: {store_name}
            - Category: {store_category}
            - Location: {store_location}
            - Average Rating: {store_average_rating}

            PRODUCTS EXPERIENCED:
            {product_details}

            USER'S PERSONAL NOTES:
            {user_note}

            REVIEW REQUIREMENTS:
            1. CONTEXT UNDERSTANDING: {store_category_guidance}
            2. If product_details is EMPTY → ONLY comment on store atmosphere, service, location, staff - DO NOT MENTION ANY SPECIFIC PRODUCTS OR MERCHANDISE
            3. If products listed → Comment on ALL products using appropriate terms for each type
            4. Match review tone to the {rating}-star rating (5=mostly positive, 1=mostly negative)
            5. Write in casual, conversational {language_name} as if texting a friend
            6. DO NOT GUESS what the store sells based on its name
            7. Your conclusion (return intention) should align with your stated rating and previous comments"

            DO NOT use generic product descriptions if no products are provided!
            """),
            # ================================ END REVIEW GENERATOR TEMPLATE ================================

        }
        
    def add_template(self, name: str, template: str, template_type: str = "text") -> None:
        """
        Add a new template or replace an existing one.
        
        Args:
            name: Template name
            template: Template string
            template_type: Type of template ('text' or 'chat')
        """
        self._templates[name] = PromptTemplate(template, template_type)

        
    def get_template(self, template_name: str) -> PromptTemplate:
        """
        Get a template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            PromptTemplate: The requested template
            
        Raises:
            ValueError: If template not found
        """
        if template_name not in self._templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self._templates[template_name]
    
    def format_messages(self, 
                        system_template: str, 
                        user_content: str, 
                        history_messages: Optional[List[Dict[str, str]]] = None, 
                        **kwargs) -> List[Dict[str, str]]:
        """
        Format a complete message list with system, history, and user messages.
        
        Args:
            system_template: Name of system template
            user_content: User message content
            history_messages: Previous messages in the conversation
            **kwargs: Variables for template formatting
            
        Returns:
            List[Dict[str, str]]: List of messages in OpenAI format
        """
        try:
            system_content = self.get_template(system_template).format(**kwargs)
            
            messages = [{"role": "system", "content": system_content}]
            
            if history_messages:
                messages.extend(history_messages)
                
            messages.append({"role": "user", "content": user_content})
            
            return messages
        except Exception as e:
            self.logger.error(f"Error formatting messages: {str(e)}")
            # Fallback to simple system message
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content}
            ]
    
    def format_rag_messages(self,
                           query: str,
                           context: str,
                           history_messages: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        Format messages specifically for RAG (Retrieval Augmented Generation).
        
        Args:
            query: User query
            context: Retrieved context
            history_messages: Previous messages in the conversation
            detailed: Whether to use detailed RAG template
            
        Returns:
            List[Dict[str, str]]: List of messages in OpenAI format
        """
        template_name = "rag_system_template"
        print(f"Using template: {template_name}")

        return self.format_messages(
            system_template=template_name,
            user_content=query,
            history_messages=history_messages,
            context=context
        )
    
    def format_react_cot_messages(self,
                                query: str,
                                context: str = "",
                                history_messages: Optional[List[Dict[str, str]]] = None,
                                enable_thinking: bool = True) -> List[Dict[str, str]]:
        """
        Format messages specifically for ReAct+CoT (Reasoning and Acting with Chain of Thought).
        
        Args:
            query: User query
            context: Retrieved context (optional)
            history_messages: Previous messages in the conversation
            
        Returns:
            List[Dict[str, str]]: List of messages in OpenAI format
        """
        template_name = "react_cot_rag_system" if context else "react_cot_system"

        thinking_instruction = "First THINK carefully before providing your answer." if enable_thinking else "DO NOT think."

        return self.format_messages(
            system_template=template_name,
            user_content=query,
            history_messages=history_messages,
            context=context,
            thinking_instruction=thinking_instruction
        )


    def format_store_review_messages(
        self,
        store_info: Dict[str, Any],
        product_items: List[Dict[str, Any]],
        rating: int,
        user_note: str = "",
        length: str = "medium",
        system_language: str = "auto"
    ) -> List[Dict[str, str]]:
        """
        Format messages specifically for store review generation.
        
        Args:
            store_info: Store information
            product_items: List of product information
            rating: Overall rating
            user_note: Additional user notes
            length: Desired review length
            system_language: Language for the review
            
        Returns:
            List[Dict[str, str]]: List of messages in OpenAI format
        """
        # Determine language instruction
        detected_lang = None
        language_code = "en"  
        language_name = "ENGLISH" # Default

        # if user_note and user_note.strip() != "":
        #     # Detect language from user note using the new method
        #     detected_lang = self.detect_language(user_note)
        #     if detected_lang:
        #         language_code = detected_lang
        # else:
        #     # Use system language
        #     if system_language and system_language.lower() != "auto":
        #         language_code = system_language.lower()

        if system_language and system_language.lower() != "auto":
            language_code = system_language.lower()

        self.logger.info(f"Detected language code: {language_code}")

        # Map language code to full name
        language_map = {
            "en": "ENGLISH",
            "vi": "VIETNAMESE",
            "fr": "FRENCH",
            "es": "SPANISH",
            "de": "GERMAN",
            "ja": "JAPANESE",
            "ko": "KOREAN",
            "zh": "CHINESE",
            "zh-cn": "CHINESE",
            "ru": "RUSSIAN",
        }

        language_name = language_map.get(language_code, "ENGLISH").upper()

        # Change from language_name to system_language
        language_instruction = f"YOU MUST WRITE THE ENTIRE REVIEW IN {language_name} LANGUAGE ONLY."

        self.logger.info(f"Language instruction: {language_instruction}")

        if length == 'long':
            review_length = "5-6"
        elif length == 'medium':
            review_length = "4-5"
        else:
            review_length = "3-4"
        self.logger.info(f"Review length {review_length}")

        # Determine if this is a product review or food review
        food_related_terms = ["food", "restaurant", "dish", "eat", "taste", "flavor", "meal", "cook"]
        is_food_related = any(term in (p.get('description', '') + " " + p.get('name', '')).lower() 
                            for p in product_items for term in food_related_terms)
        
        review_type = "restaurant/food" if is_food_related else "store/product"
        
        # Format product details
        product_details = []
        for product in product_items:
            product_info = f"- {product.get('name', '')}: {product.get('description', '')}, Price: {product.get('price', '')}\n"
            product_info += f"  Average Rating: {product.get('stats', {}).get('average_rating', 0)}, " \
                            f"Total Reviews: {product.get('stats', {}).get('total_reviews', 0)}\n"
            
            # Add reviews if available
            reviews = product.get('reviews', [])
            if reviews:
                # Classify reviews
                positive_reviews = [r for r in reviews if r.get('rating', 0) >= 4]
                negative_reviews = [r for r in reviews if r.get('rating', 0) < 3]
                
                if positive_reviews or negative_reviews:
                    product_info += "  Sample Reviews:\n"
                    
                    for review in positive_reviews:
                        product_info += f"    * {review.get('rating', 0)}★: \"{review.get('content', '')}\"\n"
                        
                    for review in negative_reviews:
                        product_info += f"    * {review.get('rating', 0)}★: \"{review.get('content', '')}\"\n"
                else:
                    product_info += "  No specific positive or negative reviews available.\n"
            else:
                product_info += "  No reviews available.\n"
                
            product_details.append(product_info)
        
        store_category = store_info.get('store_category', '')
        store_category_guidance = ""
        if store_category:
            store_category_guidance = f"Use {store_category} context to guide your review focus and vocabulary"
        else:
            store_category_guidance = "Infer context from products and experience to guide your review"

        # Format system message
        system_content = self.get_template("store_review_system").format(
            system_language=system_language,
            user_note=user_note,
            language_instruction=language_instruction,
            review_length=review_length,
            store_category_guidance=store_category_guidance,
        )
        
        # Format user message
        user_content = self.get_template("store_review_user").format(
            length=length,
            review_type=review_type,
            rating=rating,
            store_name=store_info.get('name', ''),
            store_location=store_info.get('location', ''),
            store_average_rating=store_info.get('average_rating', 0),
            store_category=store_category,
            store_category_guidance=store_category_guidance,
            product_details="".join(product_details),
            user_note=user_note,
            system_language=system_language,
            language_name=language_name,
            language_instruction=language_instruction
        )

        self.logger.info(f"Formatted system template (first 200 chars): {system_content[:200]}...")
        self.logger.info(f"Formatted user template (first 200 chars): {user_content[:200]}...")
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

# Create a singleton instance for global use
prompt_manager = PromptManager()