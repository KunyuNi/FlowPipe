import os
import torch
from typing import Dict, Optional, List
from loguru import logger

class LLMEmbeddingLoader:
    """Loader for pre-generated LLM embeddings, supporting multiple models and types."""

    def __init__(self, llm_embedding_root: str = None):
        self.llm_embedding_root = llm_embedding_root
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        # 延迟构建映射，等待实际路径设置
        self.dataset_to_embedding_map = {}

    def _build_dataset_mapping(self) -> Dict[str, str]:
        mapping = {}
        if not self.llm_embedding_root:
            return mapping
            
        try:
            # 自动添加模型和嵌入类型路径
            full_path = os.path.join(self.llm_embedding_root, "Meta-Llama-3.1-8B", "contextual_semantic")
            
            if not os.path.exists(full_path):
                logger.warning(f"LLM embedding mapping failed: path does not exist - {full_path}")
                return mapping
            
            # 扫描完整路径下的 .pt 文件
            pt_files = [f for f in os.listdir(full_path) if f.endswith('.pt')]

            for pt_file in pt_files:
                dataset_name = pt_file[:-3]  # 移除 .pt 后缀
                file_full_path = os.path.join(full_path, pt_file)
                # 使用固定的 embedding_type 和 llm_model
                key = f"{dataset_name}_contextual_semantic_Meta-Llama-3.1-8B"
                mapping[key] = file_full_path

            if len(mapping) > 0:
                logger.info(f"✓ LLM embedding mapping successful: {len(mapping)} datasets loaded")
            else:
                logger.warning(f"✗ LLM embedding mapping failed: no .pt files found in {full_path}")

        except Exception as e:
            logger.error(f"✗ LLM embedding mapping failed: {e}")

        return mapping

    def _is_pt_files_directory(self, path: str) -> bool:
        """检查目录是否直接包含 .pt 文件"""
        try:
            files = os.listdir(path)
            return any(f.endswith('.pt') for f in files)
        except:
            return False

    def _scan_pt_files_directory(self, pt_dir: str, mapping: Dict[str, str],
                                 embedding_type: str = "contextual_semantic",
                                 llm_model: str = "Meta-Llama-3.1-8B"):
        """扫描包含 .pt 文件的目录"""
        try:
            pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
            logger.debug(f"Found {len(pt_files)} .pt files in {pt_dir}")

            for pt_file in pt_files:
                dataset_name = pt_file[:-3]  # 移除 .pt 后缀
                full_path = os.path.join(pt_dir, pt_file)
                key = f"{dataset_name}_{embedding_type}_{llm_model}"
                mapping[key] = full_path
                logger.debug(f"Added mapping: {key} -> {full_path}")

        except Exception as e:
            logger.error(f"Error scanning .pt files directory {pt_dir}: {e}")
    def set_root(self, new_root: str):
        try:
            if self.llm_embedding_root != new_root:
                self.llm_embedding_root = new_root
                self.dataset_to_embedding_map = self._build_dataset_mapping()
                self.embedding_cache.clear()
        except Exception as e:
            logger.error(f"Error setting LLM embedding root to {new_root}: {e}")
    # def load_all_embeddings(self, dataset_name: str, embedding_types: List[str]) -> Optional[torch.Tensor]:
    #     combined_embeddings: List[torch.Tensor] = []
    #     cache_key = f"{dataset_name}_all"
    #     if cache_key in self.embedding_cache:
    #         return self.embedding_cache[cache_key]
    #
    #     loaded_count = 0
    #     for key, path in self.dataset_to_embedding_map.items():
    #         if dataset_name in key:  # Match dataset_name
    #             # Extract model and type from key
    #             parts = key.split('_')
    #             if len(parts) >= 3:
    #                 emb_type = parts[1]
    #                 model = parts[2]
    #                 if emb_type in embedding_types:  # Only load if type is in the required 4 styles
    #                     try:
    #                         embedding = torch.load(path)
    #                         combined_embeddings.append(embedding)
    #                         loaded_count += 1
    #                         logger.info(f"Loaded {emb_type} embedding for {dataset_name} from model {model}: {embedding.shape}")
    #                     except Exception as e:
    #                         logger.warning(f"Failed to load {key}: {e}")
    #
    #     if loaded_count == 0:
    #         logger.error(f"No embeddings loaded for {dataset_name}. Falling back required.")
    #         return None
    #
    #     # Concatenate all loaded embeddings along dimension 0
    #     combined_tensor = torch.cat(combined_embeddings, dim=0)
    #     self.embedding_cache[cache_key] = combined_tensor
    #     logger.info(f"Combined {loaded_count} embeddings (across models and types) into shape {combined_tensor.shape}")
    #     return combined_tensor

    def load_all_embeddings(self, dataset_name: str, embedding_types: List[str]) -> Optional[torch.Tensor]:
        combined_embeddings: List[torch.Tensor] = []
        cache_key = f"{dataset_name}_all"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        tried_paths = []

        def _try_load():
            loaded = 0
            for _, path in self.dataset_to_embedding_map.items():
                # 直接从路径解析
                fname = os.path.basename(path)[:-3]              # 22000-scotch-whisky-reviews
                emb_type = os.path.basename(os.path.dirname(path))        # contextual_semantic
                llm_model = os.path.basename(os.path.dirname(os.path.dirname(path)))  # Meta-Llama-3.1-8B

                if fname != dataset_name:
                    continue
                if embedding_types and emb_type not in embedding_types:
                    continue

                tried_paths.append(path)
                try:
                    emb = torch.load(path)
                    combined_embeddings.append(emb)
                    loaded += 1
                    logger.info(f"Loaded {emb_type} embedding for {dataset_name} from model {llm_model}: {emb.shape}")
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
            return loaded

        loaded_count = _try_load()
        if loaded_count == 0:
            logger.warning("No embeddings on first pass for {}. Refreshing mapping...", dataset_name)
            self.dataset_to_embedding_map = self._build_dataset_mapping()
            loaded_count = _try_load()

        if loaded_count == 0:
            logger.error(f"No embeddings loaded for {dataset_name}. Falling back required. Tried: {tried_paths}")
            return None

        combined_tensor = torch.cat(combined_embeddings, dim=0)
        self.embedding_cache[cache_key] = combined_tensor
        logger.info(f"Combined {loaded_count} embeddings into shape {combined_tensor.shape}")
        return combined_tensor



llm_embedding_loader = LLMEmbeddingLoader()




