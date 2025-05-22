"""
Agent Core with Model Orchestration Integration

This module implements the enhanced Agent Core that integrates the autonomous agent
architecture with HuggingGPT's model orchestration capabilities.
"""

import os
import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelOrchestrator:
    """
    Model Orchestrator component that manages specialized AI models.
    Inspired by HuggingGPT's approach to model selection and execution.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Model Orchestrator.
        
        Args:
            config: Configuration dictionary for the orchestrator
        """
        self.config = config or {}
        self.available_models = self._initialize_models()
        self.model_performance = {}  # Track performance metrics
        logger.info(f"Model Orchestrator initialized with {len(self.available_models)} models")
    
    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize available specialized models.
        
        Returns:
            Dictionary of available models by type
        """
        models = {}
        
        # Load models from configuration
        model_configs = self.config.get("models", {})
        
        # Initialize vision models
        vision_models = model_configs.get("vision", [])
        models["vision"] = self._load_models(vision_models, "vision")
        
        # Initialize code models
        code_models = model_configs.get("code", [])
        models["code"] = self._load_models(code_models, "code")
        
        # Initialize math models
        math_models = model_configs.get("math", [])
        models["math"] = self._load_models(math_models, "math")
        
        # Initialize audio models
        audio_models = model_configs.get("audio", [])
        models["audio"] = self._load_models(audio_models, "audio")
        
        return models
    
    def _load_models(self, model_configs: List[Dict[str, Any]], model_type: str) -> List[Dict[str, Any]]:
        """
        Load models of a specific type.
        
        Args:
            model_configs: List of model configurations
            model_type: Type of models to load
            
        Returns:
            List of initialized model objects
        """
        loaded_models = []
        
        for model_config in model_configs:
            model_id = model_config.get("id")
            model_name = model_config.get("name")
            model_endpoint = model_config.get("endpoint")
            
            try:
                # In a real implementation, this would load the actual model
                # For this example, we're just creating a placeholder
                model = {
                    "id": model_id,
                    "name": model_name,
                    "type": model_type,
                    "endpoint": model_endpoint,
                    "capabilities": model_config.get("capabilities", []),
                    "config": model_config
                }
                
                loaded_models.append(model)
                logger.info(f"Loaded {model_type} model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_type} model {model_name}: {e}")
        
        return loaded_models
    
    def select_models(self, tactical_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select appropriate models for each subtask in the tactical plan.
        
        Args:
            tactical_plan: Plan with subtasks requiring model selection
            
        Returns:
            Dictionary mapping subtask IDs to selected models
        """
        selected_models = {}
        
        for subtask in tactical_plan.get("subtasks", []):
            subtask_id = subtask.get("id")
            required_capabilities = subtask.get("required_capabilities", [])
            
            # Match required capabilities to available models
            candidate_models = self._find_candidate_models(required_capabilities)
            
            # Rank models based on performance history and requirements
            ranked_models = self._rank_models(candidate_models, subtask)
            
            # Select best model
            selected_models[subtask_id] = ranked_models[0] if ranked_models else None
            
            if ranked_models:
                logger.info(f"Selected model {ranked_models[0]['name']} for subtask {subtask_id}")
            else:
                logger.warning(f"No suitable model found for subtask {subtask_id}")
        
        return selected_models
    
    def _find_candidate_models(self, required_capabilities: List[str]) -> List[Dict[str, Any]]:
        """
        Find models that match the required capabilities.
        
        Args:
            required_capabilities: List of capabilities needed
            
        Returns:
            List of candidate models
        """
        candidates = []
        
        # Flatten all models into a single list for searching
        all_models = []
        for model_type, models in self.available_models.items():
            all_models.extend(models)
        
        # Find models that have all required capabilities
        for model in all_models:
            model_capabilities = model.get("capabilities", [])
            
            # Check if model has all required capabilities
            if all(cap in model_capabilities for cap in required_capabilities):
                candidates.append(model)
        
        return candidates
    
    def _rank_models(self, candidate_models: List[Dict[str, Any]], subtask: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank candidate models based on performance history and requirements.
        
        Args:
            candidate_models: List of candidate models
            subtask: Subtask information
            
        Returns:
            Ranked list of models
        """
        if not candidate_models:
            return []
        
        # Calculate scores for each model
        scored_models = []
        for model in candidate_models:
            model_id = model.get("id")
            
            # Get performance history for this model and task type
            task_type = subtask.get("type")
            performance = self.model_performance.get(model_id, {}).get(task_type, {})
            
            # Calculate score based on success rate and speed
            success_rate = performance.get("success_rate", 0.8)  # Default to 0.8 if no history
            avg_execution_time = performance.get("avg_execution_time", 1.0)  # Default to 1.0 if no history
            
            # Simple scoring formula: success_rate / normalized_time
            # Lower time is better, so we invert it
            normalized_time = max(0.1, min(avg_execution_time, 10.0)) / 10.0
            score = success_rate / normalized_time
            
            scored_models.append((model, score))
        
        # Sort by score in descending order
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the models, not the scores
        return [model for model, _ in scored_models]
    
    def execute_with_model(self, subtask: Dict[str, Any], model: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a subtask using the selected model.
        
        Args:
            subtask: Subtask information
            model: Selected model
            inputs: Input data for the model
            
        Returns:
            Execution results
        """
        model_id = model.get("id")
        model_type = model.get("type")
        task_type = subtask.get("type")
        
        # Prepare inputs for the specific model
        formatted_inputs = self._format_inputs_for_model(model, inputs)
        
        # Execute model
        start_time = time.time()
        try:
            # In a real implementation, this would call the actual model API
            # For this example, we're just simulating a response
            result = self._simulate_model_execution(model, formatted_inputs)
            success = True
            logger.info(f"Successfully executed {model_type} model {model_id} for task {task_type}")
        except Exception as e:
            result = {"error": str(e)}
            success = False
            logger.error(f"Failed to execute {model_type} model {model_id} for task {task_type}: {e}")
            
        execution_time = time.time() - start_time
        
        # Update performance metrics
        self._update_model_performance(model_id, task_type, success, execution_time)
        
        return {
            "result": result,
            "success": success,
            "execution_time": execution_time,
            "model": model_id,
            "task": task_type
        }
    
    def _format_inputs_for_model(self, model: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format inputs for the specific model.
        
        Args:
            model: Model information
            inputs: Raw input data
            
        Returns:
            Formatted inputs
        """
        model_type = model.get("type")
        
        # Different formatting based on model type
        if model_type == "vision":
            return self._format_vision_inputs(model, inputs)
        elif model_type == "code":
            return self._format_code_inputs(model, inputs)
        elif model_type == "math":
            return self._format_math_inputs(model, inputs)
        elif model_type == "audio":
            return self._format_audio_inputs(model, inputs)
        else:
            return inputs
    
    def _format_vision_inputs(self, model: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format inputs for vision models."""
        formatted = {}
        
        # Extract image data
        if "image" in inputs:
            formatted["image"] = inputs["image"]
        elif "image_url" in inputs:
            formatted["image_url"] = inputs["image_url"]
        
        # Add task-specific parameters
        if "task" in inputs:
            formatted["task"] = inputs["task"]
        
        return formatted
    
    def _format_code_inputs(self, model: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format inputs for code models."""
        formatted = {}
        
        # Extract code data
        if "code" in inputs:
            formatted["code"] = inputs["code"]
        
        # Add language information
        if "language" in inputs:
            formatted["language"] = inputs["language"]
        
        # Add task-specific parameters
        if "task" in inputs:
            formatted["task"] = inputs["task"]
        
        return formatted
    
    def _format_math_inputs(self, model: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format inputs for math models."""
        formatted = {}
        
        # Extract math expression
        if "expression" in inputs:
            formatted["expression"] = inputs["expression"]
        
        # Add task-specific parameters
        if "task" in inputs:
            formatted["task"] = inputs["task"]
        
        return formatted
    
    def _format_audio_inputs(self, model: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Format inputs for audio models."""
        formatted = {}
        
        # Extract audio data
        if "audio" in inputs:
            formatted["audio"] = inputs["audio"]
        elif "audio_url" in inputs:
            formatted["audio_url"] = inputs["audio_url"]
        
        # Add task-specific parameters
        if "task" in inputs:
            formatted["task"] = inputs["task"]
        
        return formatted
    
    def _simulate_model_execution(self, model: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate model execution for demonstration purposes.
        
        Args:
            model: Model information
            inputs: Formatted inputs
            
        Returns:
            Simulated results
        """
        model_type = model.get("type")
        
        # Simulate different responses based on model type
        if model_type == "vision":
            return {
                "detected_objects": ["person", "car", "tree"],
                "confidence_scores": [0.95, 0.87, 0.76]
            }
        elif model_type == "code":
            return {
                "execution_result": "Hello, World!",
                "syntax_check": "valid",
                "execution_time": 0.05
            }
        elif model_type == "math":
            return {
                "result": "42",
                "steps": ["Simplify expression", "Apply formula", "Calculate result"],
                "confidence": 0.99
            }
        elif model_type == "audio":
            return {
                "transcription": "Hello, this is a test audio.",
                "confidence": 0.92,
                "language_detected": "en"
            }
        else:
            return {"result": "Unsupported model type"}
    
    def _update_model_performance(self, model_id: str, task_type: str, success: bool, execution_time: float) -> None:
        """
        Update performance metrics for a model.
        
        Args:
            model_id: Model identifier
            task_type: Type of task
            success: Whether execution was successful
            execution_time: Execution time in seconds
        """
        # Initialize if needed
        if model_id not in self.model_performance:
            self.model_performance[model_id] = {}
        
        if task_type not in self.model_performance[model_id]:
            self.model_performance[model_id][task_type] = {
                "success_count": 0,
                "failure_count": 0,
                "total_execution_time": 0,
                "execution_count": 0,
                "success_rate": 0,
                "avg_execution_time": 0
            }
        
        # Update metrics
        metrics = self.model_performance[model_id][task_type]
        
        if success:
            metrics["success_count"] += 1
        else:
            metrics["failure_count"] += 1
        
        metrics["total_execution_time"] += execution_time
        metrics["execution_count"] += 1
        
        # Recalculate derived metrics
        total_count = metrics["success_count"] + metrics["failure_count"]
        metrics["success_rate"] = metrics["success_count"] / total_count if total_count > 0 else 0
        metrics["avg_execution_time"] = metrics["total_execution_time"] / metrics["execution_count"] if metrics["execution_count"] > 0 else 0


class HierarchicalPlanner:
    """
    Enhanced planning module that combines strategic and tactical planning.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Hierarchical Planner.
        
        Args:
            config: Configuration dictionary for the planner
        """
        self.config = config or {}
        self.llm_factory = None  # Will be set by Agent
        logger.info("Hierarchical Planner initialized")
    
    def create_strategic_plan(self, goal: str, memory: Any) -> Dict[str, Any]:
        """
        Create a high-level strategic plan to achieve the given goal.
        
        Args:
            goal: User goal
            memory: Memory module for context retrieval
            
        Returns:
            Strategic plan
        """
        # Get relevant context from memory
        context = memory.get_relevant_context(goal)
        
        # Create planning prompt
        plan_template = self._get_strategic_planning_template()
        plan_input = self._format_planning_input(goal, context)
        
        # Generate strategic plan using LLM
        llm_provider = self.llm_factory.get_active_provider()
        plan_output = llm_provider.generate(plan_template.format(input=plan_input))
        
        # Parse and structure the plan
        structured_plan = self._parse_strategic_plan(plan_output.get("text", ""))
        
        logger.info(f"Created strategic plan with {len(structured_plan.get('steps', []))} steps for goal: {goal[:50]}...")
        
        return structured_plan
    
    def create_tactical_plan(self, strategic_step: Dict[str, Any], memory: Any) -> Dict[str, Any]:
        """
        Create a tactical plan for executing a strategic step.
        
        Args:
            strategic_step: Step from the strategic plan
            memory: Memory module for context retrieval
            
        Returns:
            Tactical plan with subtasks
        """
        # Get relevant context for this specific step
        context = memory.get_relevant_context(strategic_step.get("description", ""))
        
        # Create tactical planning prompt
        tactical_template = self._get_tactical_planning_template()
        tactical_input = self._format_tactical_input(strategic_step, context)
        
        # Generate tactical plan using LLM
        llm_provider = self.llm_factory.get_active_provider()
        tactical_output = llm_provider.generate(tactical_template.format(input=tactical_input))
        
        # Parse and structure the tactical plan
        tactical_plan = self._parse_tactical_plan(tactical_output.get("text", ""), strategic_step)
        
        # Identify required capabilities for each subtask
        for subtask in tactical_plan.get("subtasks", []):
            subtask["required_capabilities"] = self._identify_capabilities(subtask)
        
        logger.info(f"Created tactical plan with {len(tactical_plan.get('subtasks', []))} subtasks for step: {strategic_step.get('description', '')[:50]}...")
        
        return tactical_plan
    
    def _get_strategic_planning_template(self) -> str:
        """Get the template for strategic planning."""
        return """
        You are an AI assistant tasked with creating a strategic plan to achieve a user's goal.
        Break down the goal into logical steps that can be executed sequentially.
        
        USER GOAL: {input}
        
        Create a strategic plan with 3-7 steps. For each step, provide:
        1. A clear description of what needs to be done
        2. The expected outcome of the step
        3. Any dependencies on previous steps
        
        Format your response as a numbered list of steps.
        """
    
    def _get_tactical_planning_template(self) -> str:
        """Get the template for tactical planning."""
        return """
        You are an AI assistant tasked with breaking down a strategic step into specific subtasks
        that can be executed by specialized AI models.
        
        STRATEGIC STEP: {input}
        
        Break this step down into 2-5 subtasks. For each subtask, specify:
        1. A clear description of the subtask
        2. The type of task (vision, code, math, text, audio)
        3. The expected inputs and outputs
        
        Format your response as a numbered list of subtasks.
        """
    
    def _format_planning_input(self, goal: str, context: str) -> str:
        """Format the input for strategic planning."""
        return f"""
        GOAL: {goal}
        
        RELEVANT CONTEXT:
        {context}
        """
    
    def _format_tactical_input(self, strategic_step: Dict[str, Any], context: str) -> str:
        """Format the input for tactical planning."""
        return f"""
        STEP DESCRIPTION: {strategic_step.get('description', '')}
        EXPECTED OUTCOME: {strategic_step.get('expected_outcome', '')}
        
        RELEVANT CONTEXT:
        {context}
        """
    
    def _parse_strategic_plan(self, plan_text: str) -> Dict[str, Any]:
        """
        Parse the LLM output into a structured strategic plan.
        
        Args:
            plan_text: Raw text output from LLM
            
        Returns:
            Structured strategic plan
        """
        # In a real implementation, this would use regex or more sophisticated parsing
        # For this example, we'll create a simple structured plan
        
        lines = plan_text.strip().split('\n')
        steps = []
        
        current_step = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a new step
            if line[0].isdigit() and '.' in line[:5]:
                if current_step:
                    steps.append(current_step)
                
                step_text = line.split('.', 1)[1].strip()
                current_step = {
                    "id": len(steps) + 1,
                    "description": step_text,
                    "expected_outcome": "",
                    "dependencies": []
                }
            elif current_step and line.lower().startswith("outcome:"):
                current_step["expected_outcome"] = line.split(":", 1)[1].strip()
            elif current_step:
                # Append to the current step description
                current_step["description"] += " " + line
        
        # Add the last step if it exists
        if current_step:
            steps.append(current_step)
        
        return {
            "steps": steps,
            "goal": plan_text.split('\n', 1)[0] if lines else ""
        }
    
    def _parse_tactical_plan(self, plan_text: str, strategic_step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the LLM output into a structured tactical plan.
        
        Args:
            plan_text: Raw text output from LLM
            strategic_step: The strategic step this tactical plan is for
            
        Returns:
            Structured tactical plan with subtasks
        """
        lines = plan_text.strip().split('\n')
        subtasks = []
        
        current_subtask = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a new subtask
            if line[0].isdigit() and '.' in line[:5]:
                if current_subtask:
                    subtasks.append(current_subtask)
                
                subtask_text = line.split('.', 1)[1].strip()
                current_subtask = {
                    "id": f"subtask_{strategic_step.get('id', 0)}_{len(subtasks) + 1}",
                    "description": subtask_text,
                    "type": "text",  # Default type
                    "inputs": {},
                    "expected_outputs": {}
                }
            elif current_subtask and "type:" in line.lower():
                task_type = line.lower().split("type:", 1)[1].strip()
                # Map to one of our supported types
                if "vision" in task_type or "image" in task_type:
                    current_subtask["type"] = "vision"
                elif "code" in task_type or "program" in task_type:
                    current_subtask["type"] = "code"
                elif "math" in task_type or "calculation" in task_type:
                    current_subtask["type"] = "math"
                elif "audio" in task_type or "speech" in task_type:
                    current_subtask["type"] = "audio"
                else:
                    current_subtask["type"] = "text"
            elif current_subtask and "input:" in line.lower():
                input_desc = line.split("input:", 1)[1].strip()
                current_subtask["inputs"]["description"] = input_desc
            elif current_subtask and "output:" in line.lower():
                output_desc = line.split("output:", 1)[1].strip()
                current_subtask["expected_outputs"]["description"] = output_desc
            elif current_subtask:
                # Append to the current subtask description
                current_subtask["description"] += " " + line
        
        # Add the last subtask if it exists
        if current_subtask:
            subtasks.append(current_subtask)
        
        return {
            "strategic_step_id": strategic_step.get("id", 0),
            "subtasks": subtasks
        }
    
    def _identify_capabilities(self, subtask: Dict[str, Any]) -> List[str]:
        """
        Identify required capabilities for a subtask.
        
        Args:
            subtask: Subtask information
            
        Returns:
            List of required capabilities
        """
        task_type = subtask.get("type", "")
        description = subtask.get("description", "").lower()
        
        capabilities = []
        
        # Add basic capability based on task type
        if task_type == "vision":
            capabilities.append("image_understanding")
            
            # Add more specific capabilities based on description
            if "detect" in description or "object" in description:
                capabilities.append("object_detection")
            if "classify" in description:
                capabilities.append("image_classification")
            if "segment" in description:
                capabilities.append("image_segmentation")
            if "caption" in description or "describe" in description:
                capabilities.append("image_captioning")
                
        elif task_type == "code":
            capabilities.append("code_processing")
            
            # Add more specific capabilities based on description
            if "execute" in description or "run" in description:
                capabilities.append("code_execution")
            if "generate" in description or "create" in description:
                capabilities.append("code_generation")
            if "complete" in description:
                capabilities.append("code_completion")
            if "debug" in description or "fix" in description:
                capabilities.append("code_debugging")
                
        elif task_type == "math":
            capabilities.append("mathematical_reasoning")
            
            # Add more specific capabilities based on description
            if "equation" in description or "solve" in description:
                capabilities.append("equation_solving")
            if "statistic" in description or "probability" in description:
                capabilities.append("statistical_analysis")
            if "plot" in description or "graph" in description:
                capabilities.append("mathematical_plotting")
                
        elif task_type == "audio":
            capabilities.append("audio_processing")
            
            # Add more specific capabilities based on description
            if "transcribe" in description or "speech to text" in description:
                capabilities.append("speech_to_text")
            if "recognize" in description or "identify" in description:
                capabilities.append("audio_recognition")
            if "generate" in description or "synthesize" in description:
                capabilities.append("speech_synthesis")
        
        return capabilities


class MultimodalMemory:
    """
    Enhanced memory module that supports multiple modalities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Multimodal Memory.
        
        Args:
            config: Configuration dictionary for memory
        """
        self.config = config or {}
        self.short_term = {}  # In-memory storage
        self.vector_db = self._initialize_vector_db()
        self.image_store = self._initialize_image_store()
        self.code_store = self._initialize_code_store()
        logger.info("Multimodal Memory initialized")
    
    def _initialize_vector_db(self) -> Dict[str, Any]:
        """Initialize vector database for text storage."""
        # In a real implementation, this would connect to a vector database
        # For this example, we'll use a simple in-memory structure
        return {
            "vectors": [],
            "metadata": []
        }
    
    def _initialize_image_store(self) -> Dict[str, Any]:
        """Initialize storage for images."""
        return {
            "images": [],
            "embeddings": [],
            "metadata": []
        }
    
    def _initialize_code_store(self) -> Dict[str, Any]:
        """Initialize storage for code snippets."""
        return {
            "code": [],
            "execution_results": [],
            "metadata": []
        }
    
    def add(self, category: str, content: str) -> None:
        """
        Add text content to memory.
        
        Args:
            category: Category label for the content
            content: Text content to store
        """
        # Add to short-term memory
        if category not in self.short_term:
            self.short_term[category] = []
        self.short_term[category].append(content)
        
        # Add to vector database for long-term storage
        embedding = self._generate_text_embedding(content)
        self.vector_db["vectors"].append(embedding)
        self.vector_db["metadata"].append({
            "category": category,
            "content": content,
            "timestamp": time.time()
        })
        
        logger.info(f"Added content to memory category: {category}")
    
    def add_multimodal(self, category: str, content: Dict[str, Any]) -> None:
        """
        Add multimodal content to memory.
        
        Args:
            category: Category label for the content
            content: Dictionary with different modality content
        """
        # Extract and store text components
        if "text" in content:
            self.add(f"text_{category}", content["text"])
            
        # Store images with embeddings
        if "images" in content:
            for image in content["images"]:
                image_embedding = self._generate_image_embedding(image)
                self.image_store["images"].append(image)
                self.image_store["embeddings"].append(image_embedding)
                self.image_store["metadata"].append({
                    "category": category,
                    "timestamp": time.time()
                })
                
        # Store code with execution results
        if "code" in content:
            self.code_store["code"].append(content["code"])
            self.code_store["execution_results"].append(content.get("execution_result"))
            self.code_store["metadata"].append({
                "category": category,
                "language": content.get("language", "unknown"),
                "timestamp": time.time()
            })
        
        logger.info(f"Added multimodal content to memory category: {category}")
    
    def get_relevant_context(self, query: str, limit: int = 5) -> str:
        """
        Retrieve relevant memories based on a query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Formatted context string
        """
        # Get text results from vector database
        query_embedding = self._generate_text_embedding(query)
        text_results = self._query_vector_db(query_embedding, limit)
        
        # Format results into a context string
        context_parts = []
        
        for i, result in enumerate(text_results):
            content = result.get("content", "")
            category = result.get("category", "Unknown")
            
            context_parts.append(f"[{i+1}] {content}\nCategory: {category}\n")
        
        # Add recent items from short-term memory
        for category, items in self.short_term.items():
            for item in items[-2:]:  # Last 2 items from each category
                if item not in context_parts:  # Avoid duplicates
                    context_parts.append(f"[Recent {category}] {item}\n")
        
        return "\n".join(context_parts)
    
    def retrieve_multimodal(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant multimodal memories.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Dictionary with different modality results
        """
        # Get text results
        query_embedding = self._generate_text_embedding(query)
        text_results = self._query_vector_db(query_embedding, limit)
        
        # Get image results using multimodal embedding
        image_results = self._query_image_store(query, limit=3)
        
        # Get code results
        code_results = self._query_code_store(query, limit=2)
        
        # Combine results
        return {
            "text": text_results,
            "images": image_results,
            "code": code_results
        }
    
    def _generate_text_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # In a real implementation, this would use a text embedding model
        # For this example, we'll create a simple mock embedding
        import hashlib
        import struct
        
        # Create a deterministic but unique representation based on the text
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash bytes to a list of floats
        floats = []
        for i in range(0, len(hash_bytes), 4):
            if i + 4 <= len(hash_bytes):
                float_val = struct.unpack('f', hash_bytes[i:i+4])[0]
                floats.append(float_val)
        
        # Normalize to unit length
        magnitude = sum(x*x for x in floats) ** 0.5
        if magnitude > 0:
            floats = [x/magnitude for x in floats]
        
        return floats
    
    def _generate_image_embedding(self, image: Any) -> List[float]:
        """
        Generate embedding vector for an image.
        
        Args:
            image: Input image
            
        Returns:
            Embedding vector
        """
        # In a real implementation, this would use an image embedding model
        # For this example, we'll create a simple mock embedding
        import hashlib
        import struct
        
        # Create a deterministic but unique representation
        # In a real system, this would be based on image content
        hash_obj = hashlib.md5(str(id(image)).encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash bytes to a list of floats
        floats = []
        for i in range(0, len(hash_bytes), 4):
            if i + 4 <= len(hash_bytes):
                float_val = struct.unpack('f', hash_bytes[i:i+4])[0]
                floats.append(float_val)
        
        # Normalize to unit length
        magnitude = sum(x*x for x in floats) ** 0.5
        if magnitude > 0:
            floats = [x/magnitude for x in floats]
        
        return floats
    
    def _query_vector_db(self, query_embedding: List[float], limit: int) -> List[Dict[str, Any]]:
        """
        Query the vector database for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            
        Returns:
            List of matching items with metadata
        """
        # In a real implementation, this would perform a similarity search
        # For this example, we'll return the most recent items
        
        # Sort by recency (most recent first)
        sorted_results = sorted(
            self.vector_db["metadata"],
            key=lambda x: x.get("timestamp", 0),
            reverse=True
        )
        
        return sorted_results[:limit]
    
    def _query_image_store(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Query the image store.
        
        Args:
            query: Text query
            limit: Maximum number of results
            
        Returns:
            List of matching images with metadata
        """
        # In a real implementation, this would perform a multimodal search
        # For this example, we'll return the most recent images
        
        results = []
        for i in range(min(limit, len(self.image_store["images"]))):
            results.append({
                "image": self.image_store["images"][i],
                "metadata": self.image_store["metadata"][i]
            })
        
        return results
    
    def _query_code_store(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Query the code store.
        
        Args:
            query: Text query
            limit: Maximum number of results
            
        Returns:
            List of matching code snippets with metadata
        """
        # In a real implementation, this would perform a code-aware search
        # For this example, we'll return the most recent code snippets
        
        results = []
        for i in range(min(limit, len(self.code_store["code"]))):
            results.append({
                "code": self.code_store["code"][i],
                "execution_result": self.code_store["execution_results"][i],
                "metadata": self.code_store["metadata"][i]
            })
        
        return results


class EnhancedDecisionModule:
    """
    Enhanced decision module that combines strategic decision-making with model selection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Enhanced Decision Module.
        
        Args:
            config: Configuration dictionary for the decision module
        """
        self.config = config or {}
        self.llm_factory = None  # Will be set by Agent
        logger.info("Enhanced Decision Module initialized")
    
    def make_strategic_decision(self, options: List[Dict[str, Any]], context: str, goal: str) -> Dict[str, Any]:
        """
        Make a high-level strategic decision.
        
        Args:
            options: List of available options
            context: Relevant context
            goal: User goal
            
        Returns:
            Selected option
        """
        # Create decision prompt
        decision_template = self._get_strategic_decision_template()
        decision_input = self._format_strategic_input(options, context, goal)
        
        # Generate decision using LLM
        llm_provider = self.llm_factory.get_active_provider()
        decision_output = llm_provider.generate(decision_template.format(input=decision_input))
        
        # Parse and extract the decision
        selected_option = self._parse_strategic_decision(decision_output.get("text", ""), options)
        
        logger.info(f"Made strategic decision: {selected_option.get('description', '')[:50]}...")
        
        return selected_option
    
    def select_models_for_plan(self, tactical_plan: Dict[str, Any], model_orchestrator: Any) -> Dict[str, Any]:
        """
        Select appropriate models for a tactical plan.
        
        Args:
            tactical_plan: Tactical plan with subtasks
            model_orchestrator: Model orchestrator component
            
        Returns:
            Dictionary mapping subtasks to selected models
        """
        # Delegate to the model orchestrator
        selected_models = model_orchestrator.select_models(tactical_plan)
        
        # Log the selections
        for subtask_id, model in selected_models.items():
            if model:
                logger.info(f"Selected model {model.get('name', 'unknown')} for subtask {subtask_id}")
            else:
                logger.warning(f"No suitable model found for subtask {subtask_id}")
        
        return selected_models
    
    def evaluate_execution_results(self, results: List[Dict[str, Any]], expected_outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the results of execution against expected outcomes.
        
        Args:
            results: Execution results
            expected_outcomes: Expected outcomes
            
        Returns:
            Evaluation results
        """
        evaluation = {
            "success": True,
            "partial_success": False,
            "failure_reasons": [],
            "subtask_evaluations": {}
        }
        
        # Evaluate each subtask result
        for result in results:
            subtask_id = result.get("subtask_id")
            success = result.get("success", False)
            
            subtask_eval = {
                "success": success,
                "execution_time": result.get("execution_time", 0),
                "model_used": result.get("model", "unknown")
            }
            
            if not success:
                subtask_eval["failure_reason"] = result.get("error", "Unknown error")
                evaluation["failure_reasons"].append(f"Subtask {subtask_id}: {subtask_eval['failure_reason']}")
                evaluation["success"] = False
                evaluation["partial_success"] = True
            
            evaluation["subtask_evaluations"][subtask_id] = subtask_eval
        
        return evaluation
    
    def _get_strategic_decision_template(self) -> str:
        """Get the template for strategic decision-making."""
        return """
        You are an AI assistant tasked with making a strategic decision based on multiple options.
        Evaluate each option against the user's goal and the provided context.
        
        {input}
        
        Select the best option and explain your reasoning.
        Format your response as:
        SELECTED OPTION: [option number]
        REASONING: [your explanation]
        """
    
    def _format_strategic_input(self, options: List[Dict[str, Any]], context: str, goal: str) -> str:
        """Format the input for strategic decision-making."""
        options_text = ""
        for i, option in enumerate(options):
            options_text += f"OPTION {i+1}: {option.get('description', '')}\n"
        
        return f"""
        USER GOAL: {goal}
        
        AVAILABLE OPTIONS:
        {options_text}
        
        RELEVANT CONTEXT:
        {context}
        """
    
    def _parse_strategic_decision(self, decision_text: str, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse the LLM output to extract the selected option.
        
        Args:
            decision_text: Raw text output from LLM
            options: Available options
            
        Returns:
            Selected option
        """
        # Look for "SELECTED OPTION: [number]" pattern
        import re
        
        option_match = re.search(r"SELECTED OPTION:\s*(\d+)", decision_text)
        if option_match:
            option_num = int(option_match.group(1))
            if 1 <= option_num <= len(options):
                return options[option_num - 1]
        
        # If no clear selection found, default to the first option
        return options[0] if options else {}


class ComprehensiveMonitor:
    """
    Enhanced monitoring module that tracks both system and model performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Comprehensive Monitor.
        
        Args:
            config: Configuration dictionary for the monitor
        """
        self.config = config or {}
        self.system_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0,
            "average_execution_time": 0
        }
        self.model_metrics = {}
        self.execution_history = []
        logger.info("Comprehensive Monitor initialized")
    
    def evaluate_execution(self, plan: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the execution of a plan.
        
        Args:
            plan: The executed plan
            results: Execution results
            
        Returns:
            Evaluation metrics
        """
        # Update system metrics
        self.system_metrics["total_executions"] += 1
        
        execution_time = results.get("execution_time", 0)
        self.system_metrics["total_execution_time"] += execution_time
        
        success = results.get("success", False)
        if success:
            self.system_metrics["successful_executions"] += 1
        else:
            self.system_metrics["failed_executions"] += 1
        
        # Calculate average execution time
        if self.system_metrics["total_executions"] > 0:
            self.system_metrics["average_execution_time"] = (
                self.system_metrics["total_execution_time"] / 
                self.system_metrics["total_executions"]
            )
        
        # Update model metrics
        for subtask_id, subtask_result in results.get("subtask_results", {}).items():
            model_id = subtask_result.get("model", "unknown")
            
            if model_id not in self.model_metrics:
                self.model_metrics[model_id] = {
                    "total_uses": 0,
                    "successful_uses": 0,
                    "failed_uses": 0,
                    "total_execution_time": 0,
                    "average_execution_time": 0
                }
            
            model_metrics = self.model_metrics[model_id]
            model_metrics["total_uses"] += 1
            
            subtask_success = subtask_result.get("success", False)
            if subtask_success:
                model_metrics["successful_uses"] += 1
            else:
                model_metrics["failed_uses"] += 1
            
            subtask_time = subtask_result.get("execution_time", 0)
            model_metrics["total_execution_time"] += subtask_time
            
            if model_metrics["total_uses"] > 0:
                model_metrics["average_execution_time"] = (
                    model_metrics["total_execution_time"] / 
                    model_metrics["total_uses"]
                )
        
        # Record execution in history
        execution_record = {
            "timestamp": time.time(),
            "plan": plan,
            "results": results,
            "success": success,
            "execution_time": execution_time
        }
        self.execution_history.append(execution_record)
        
        # Limit history size
        max_history = self.config.get("max_history_size", 100)
        if len(self.execution_history) > max_history:
            self.execution_history = self.execution_history[-max_history:]
        
        logger.info(f"Evaluated execution: success={success}, time={execution_time:.2f}s")
        
        # Return current metrics
        return {
            "system_metrics": self.system_metrics,
            "model_metrics": self.model_metrics,
            "execution_success": success,
            "execution_time": execution_time
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health metrics.
        
        Returns:
            System health metrics
        """
        # Calculate success rate
        success_rate = 0
        if self.system_metrics["total_executions"] > 0:
            success_rate = (
                self.system_metrics["successful_executions"] / 
                self.system_metrics["total_executions"]
            )
        
        # Determine system health status
        status = "healthy"
        if success_rate < 0.5:
            status = "critical"
        elif success_rate < 0.8:
            status = "degraded"
        
        return {
            "status": status,
            "success_rate": success_rate,
            "average_execution_time": self.system_metrics["average_execution_time"],
            "total_executions": self.system_metrics["total_executions"],
            "timestamp": time.time()
        }
    
    def get_model_performance(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for models.
        
        Args:
            model_id: Optional specific model ID
            
        Returns:
            Model performance metrics
        """
        if model_id:
            # Return metrics for specific model
            if model_id in self.model_metrics:
                metrics = self.model_metrics[model_id]
                
                # Calculate success rate
                success_rate = 0
                if metrics["total_uses"] > 0:
                    success_rate = metrics["successful_uses"] / metrics["total_uses"]
                
                return {
                    "model_id": model_id,
                    "success_rate": success_rate,
                    "average_execution_time": metrics["average_execution_time"],
                    "total_uses": metrics["total_uses"]
                }
            else:
                return {"error": f"No metrics available for model {model_id}"}
        else:
            # Return summary metrics for all models
            model_summary = {}
            for model_id, metrics in self.model_metrics.items():
                success_rate = 0
                if metrics["total_uses"] > 0:
                    success_rate = metrics["successful_uses"] / metrics["total_uses"]
                
                model_summary[model_id] = {
                    "success_rate": success_rate,
                    "average_execution_time": metrics["average_execution_time"],
                    "total_uses": metrics["total_uses"]
                }
            
            return model_summary
    
    def get_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent execution records.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of recent execution records
        """
        # Return the most recent executions
        recent = self.execution_history[-limit:] if self.execution_history else []
        
        # Simplify the records for readability
        simplified = []
        for record in recent:
            simplified.append({
                "timestamp": record["timestamp"],
                "success": record["success"],
                "execution_time": record["execution_time"],
                "plan_steps": len(record.get("plan", {}).get("steps", [])),
                "failure_reasons": record.get("results", {}).get("failure_reasons", [])
            })
        
        return simplified


class Agent:
    """
    Enhanced Agent Core with Model Orchestration Integration.
    Combines the autonomous agent architecture with HuggingGPT's model orchestration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Agent.
        
        Args:
            config: Configuration dictionary for the agent
        """
        self.config = config or self._load_default_config()
        
        # Initialize components
        self.planner = HierarchicalPlanner(self.config.get("planning"))
        self.memory = MultimodalMemory(self.config.get("memory"))
        self.decision = EnhancedDecisionModule(self.config.get("decision"))
        self.monitor = ComprehensiveMonitor(self.config.get("monitoring"))
        self.model_orchestrator = ModelOrchestrator(self.config.get("models"))
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Set current goal
        self.current_goal = None
        
        logger.info("Agent initialized with model orchestration capabilities")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "planning": {
                "max_steps": 5,
                "planning_model": "gpt-4"
            },
            "memory": {
                "short_term_capacity": 10,
                "long_term_enabled": True,
                "vector_db": "weaviate"
            },
            "decision": {
                "model": "gpt-4",
                "temperature": 0.2
            },
            "monitoring": {
                "log_level": "info",
                "metrics_enabled": True,
                "alert_threshold": 0.8,
                "max_history_size": 100
            },
            "models": {
                "vision": [
                    {
                        "id": "vision_model_1",
                        "name": "CLIP",
                        "endpoint": "https://api.example.com/models/clip",
                        "capabilities": ["image_understanding", "object_detection", "image_classification"]
                    }
                ],
                "code": [
                    {
                        "id": "code_model_1",
                        "name": "CodeExecutor",
                        "endpoint": "https://api.example.com/models/code-executor",
                        "capabilities": ["code_processing", "code_execution", "code_generation"]
                    }
                ],
                "math": [
                    {
                        "id": "math_model_1",
                        "name": "MathSolver",
                        "endpoint": "https://api.example.com/models/math-solver",
                        "capabilities": ["mathematical_reasoning", "equation_solving", "statistical_analysis"]
                    }
                ],
                "audio": [
                    {
                        "id": "audio_model_1",
                        "name": "Whisper",
                        "endpoint": "https://api.example.com/models/whisper",
                        "capabilities": ["audio_processing", "speech_to_text", "audio_recognition"]
                    }
                ]
            }
        }
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools."""
        # In a real implementation, this would set up various tools
        # For this example, we'll return a simple placeholder
        return {
            "web_search": {
                "enabled": True,
                "endpoint": "https://api.example.com/search"
            },
            "calculator": {
                "enabled": True
            },
            "document_retrieval": {
                "enabled": True,
                "source": "weaviate"
            }
        }
    
    def set_llm_factory(self, llm_factory: Any) -> None:
        """
        Set the LLM provider factory for the agent.
        
        Args:
            llm_factory: LLM provider factory
        """
        self.llm_factory = llm_factory
        self.planner.llm_factory = llm_factory
        self.decision.llm_factory = llm_factory
        logger.info("LLM factory set for agent and components")
    
    def set_goal(self, goal: str) -> None:
        """
        Set the agent's current goal.
        
        Args:
            goal: User goal
        """
        self.current_goal = goal
        self.memory.add("goals", goal)
        logger.info(f"Goal set: {goal[:50]}...")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the current goal using the enhanced architecture.
        
        Returns:
            Execution results
        """
        if not self.current_goal:
            return {"error": "No goal set"}
        
        start_time = time.time()
        
        try:
            # Create strategic plan
            strategic_plan = self.planner.create_strategic_plan(self.current_goal, self.memory)
            logger.info(f"Created strategic plan with {len(strategic_plan.get('steps', []))} steps")
            
            # Execute strategic plan
            results = self._execute_strategic_plan(strategic_plan)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Add execution time to results
            results["execution_time"] = execution_time
            
            # Evaluate execution
            evaluation = self.monitor.evaluate_execution(strategic_plan, results)
            
            # Add results to memory
            self.memory.add("execution_results", json.dumps(results))
            
            logger.info(f"Goal execution completed in {execution_time:.2f}s with success={results.get('success', False)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing goal: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _execute_strategic_plan(self, strategic_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a strategic plan.
        
        Args:
            strategic_plan: Strategic plan to execute
            
        Returns:
            Execution results
        """
        results = {
            "success": True,
            "step_results": {},
            "failure_reasons": []
        }
        
        # Execute each step in the plan
        for step in strategic_plan.get("steps", []):
            step_id = step.get("id")
            
            logger.info(f"Executing strategic step {step_id}: {step.get('description', '')[:50]}...")
            
            # Create tactical plan for this strategic step
            tactical_plan = self.planner.create_tactical_plan(step, self.memory)
            
            # Select models for the tactical plan
            selected_models = self.decision.select_models_for_plan(tactical_plan, self.model_orchestrator)
            
            # Execute the tactical plan
            step_result = self._execute_tactical_plan(tactical_plan, selected_models)
            
            # Store the result
            results["step_results"][step_id] = step_result
            
            # Check for failure
            if not step_result.get("success", False):
                results["success"] = False
                failure_reason = f"Step {step_id} failed: {step_result.get('error', 'Unknown error')}"
                results["failure_reasons"].append(failure_reason)
                
                # Decide whether to continue or abort
                if not self._should_continue_after_failure(strategic_plan, results, step_id):
                    logger.warning(f"Aborting plan execution after step {step_id} failure")
                    break
        
        return results
    
    def _execute_tactical_plan(self, tactical_plan: Dict[str, Any], selected_models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tactical plan using selected models.
        
        Args:
            tactical_plan: Tactical plan to execute
            selected_models: Models selected for each subtask
            
        Returns:
            Execution results
        """
        results = {
            "success": True,
            "subtask_results": {},
            "failure_reasons": []
        }
        
        # Execute each subtask in the plan
        for subtask in tactical_plan.get("subtasks", []):
            subtask_id = subtask.get("id")
            model = selected_models.get(subtask_id)
            
            if not model:
                # No suitable model found for this subtask
                error = f"No suitable model found for subtask {subtask_id}"
                results["subtask_results"][subtask_id] = {
                    "success": False,
                    "error": error,
                    "subtask_id": subtask_id
                }
                results["success"] = False
                results["failure_reasons"].append(error)
                continue
            
            logger.info(f"Executing subtask {subtask_id} with model {model.get('name', 'unknown')}")
            
            # Prepare inputs for the subtask
            inputs = self._prepare_subtask_inputs(subtask)
            
            # Execute the subtask with the selected model
            subtask_result = self.model_orchestrator.execute_with_model(subtask, model, inputs)
            
            # Add subtask ID to the result
            subtask_result["subtask_id"] = subtask_id
            
            # Store the result
            results["subtask_results"][subtask_id] = subtask_result
            
            # Check for failure
            if not subtask_result.get("success", False):
                results["success"] = False
                failure_reason = f"Subtask {subtask_id} failed: {subtask_result.get('error', 'Unknown error')}"
                results["failure_reasons"].append(failure_reason)
            
            # Store results in memory if successful
            if subtask_result.get("success", False):
                result_data = subtask_result.get("result", {})
                
                # Store different types of results based on subtask type
                if subtask.get("type") == "vision":
                    self.memory.add_multimodal("vision_results", {
                        "text": json.dumps(result_data),
                        "images": inputs.get("image", [])
                    })
                elif subtask.get("type") == "code":
                    self.memory.add_multimodal("code_results", {
                        "code": inputs.get("code", ""),
                        "execution_result": result_data.get("execution_result", ""),
                        "language": inputs.get("language", "unknown")
                    })
                elif subtask.get("type") == "audio":
                    self.memory.add_multimodal("audio_results", {
                        "text": result_data.get("transcription", ""),
                        "audio": inputs.get("audio", None)
                    })
                else:
                    # Default to storing as text
                    self.memory.add("subtask_results", json.dumps(result_data))
        
        return results
    
    def _prepare_subtask_inputs(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare inputs for a subtask.
        
        Args:
            subtask: Subtask information
            
        Returns:
            Prepared inputs
        """
        subtask_type = subtask.get("type", "")
        description = subtask.get("description", "")
        
        # Basic inputs that apply to all subtasks
        inputs = {
            "task": description,
            "type": subtask_type
        }
        
        # Add type-specific inputs
        if subtask_type == "vision":
            # In a real implementation, this would retrieve or generate an image
            # For this example, we'll use a placeholder
            inputs["image"] = "placeholder_image_data"
            
        elif subtask_type == "code":
            # Extract code from description or use a placeholder
            code_match = None
            if "```" in description:
                # Try to extract code blocks
                import re
                code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", description, re.DOTALL)
                if code_blocks:
                    inputs["code"] = code_blocks[0]
                    
                    # Try to determine language
                    lang_match = re.search(r"```(\w+)\n", description)
                    if lang_match:
                        inputs["language"] = lang_match.group(1)
                    else:
                        inputs["language"] = "python"  # Default to Python
            else:
                # No code block found, use a placeholder
                inputs["code"] = "print('Hello, World!')"
                inputs["language"] = "python"
                
        elif subtask_type == "math":
            # Extract math expression from description or use a placeholder
            import re
            math_expr = re.search(r"\$(.*?)\$", description)
            if math_expr:
                inputs["expression"] = math_expr.group(1)
            else:
                # No math expression found, use the description as the expression
                inputs["expression"] = description
                
        elif subtask_type == "audio":
            # In a real implementation, this would retrieve or generate audio
            # For this example, we'll use a placeholder
            inputs["audio"] = "placeholder_audio_data"
        
        return inputs
    
    def _should_continue_after_failure(self, plan: Dict[str, Any], results: Dict[str, Any], failed_step_id: Any) -> bool:
        """
        Decide whether to continue plan execution after a step failure.
        
        Args:
            plan: The strategic plan
            results: Current execution results
            failed_step_id: ID of the failed step
            
        Returns:
            True if execution should continue, False otherwise
        """
        # Get the failed step
        failed_step = None
        for step in plan.get("steps", []):
            if step.get("id") == failed_step_id:
                failed_step = step
                break
        
        if not failed_step:
            return False
        
        # Check if this step is critical (has dependents)
        has_dependents = False
        for step in plan.get("steps", []):
            if failed_step_id in step.get("dependencies", []):
                has_dependents = True
                break
        
        # If the step has dependents, we should not continue
        if has_dependents:
            return False
        
        # Check failure severity
        failure_count = len(results.get("failure_reasons", []))
        max_failures = self.config.get("planning", {}).get("max_failures", 2)
        
        # If we've exceeded the maximum number of allowed failures, stop
        if failure_count > max_failures:
            return False
        
        # Otherwise, continue execution
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            Status information
        """
        return {
            "current_goal": self.current_goal,
            "system_health": self.monitor.get_system_health(),
            "model_performance": self.monitor.get_model_performance(),
            "recent_executions": self.monitor.get_recent_executions(5)
        }
    
    def reset(self) -> None:
        """Reset the agent state."""
        self.current_goal = None
        self.memory.short_term = {}
        logger.info("Agent state reset")


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = {
        "planning": {
            "max_steps": 5,
            "planning_model": "gpt-4",
            "max_failures": 2
        },
        "memory": {
            "short_term_capacity": 10,
            "long_term_enabled": True,
            "vector_db": "weaviate"
        },
        "decision": {
            "model": "gpt-4",
            "temperature": 0.2
        },
        "monitoring": {
            "log_level": "info",
            "metrics_enabled": True,
            "alert_threshold": 0.8
        },
        "models": {
            "vision": [
                {
                    "id": "vision_model_1",
                    "name": "CLIP",
                    "endpoint": "https://api.example.com/models/clip",
                    "capabilities": ["image_understanding", "object_detection", "image_classification"]
                }
            ],
            "code": [
                {
                    "id": "code_model_1",
                    "name": "CodeExecutor",
                    "endpoint": "https://api.example.com/models/code-executor",
                    "capabilities": ["code_processing", "code_execution", "code_generation"]
                }
            ],
            "math": [
                {
                    "id": "math_model_1",
                    "name": "MathSolver",
                    "endpoint": "https://api.example.com/models/math-solver",
                    "capabilities": ["mathematical_reasoning", "equation_solving", "statistical_analysis"]
                }
            ],
            "audio": [
                {
                    "id": "audio_model_1",
                    "name": "Whisper",
                    "endpoint": "https://api.example.com/models/whisper",
                    "capabilities": ["audio_processing", "speech_to_text", "audio_recognition"]
                }
            ]
        }
    }
    
    # Initialize agent
    agent = Agent(config)
    
    # Set a goal
    agent.set_goal("Research and summarize Chapter 3 of Ruppert's book on statistical models")
    
    # In a real implementation, you would set the LLM factory here
    # agent.set_llm_factory(llm_factory)
    
    # Execute the goal
    # results = agent.execute()
    
    # Print status
    print(agent.get_status())
