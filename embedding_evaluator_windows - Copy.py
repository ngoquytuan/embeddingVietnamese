# embedding_evaluator_windows.py - Phiên bản Windows-friendly
import json
import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VietnameseSentenceSplitter:
    """
    Tách câu tiếng Việt bằng regex - thay thế cho underthesea
    """
    def __init__(self):
        # Pattern để tách câu tiếng Việt
        self.sentence_endings = r'[.!?…]\s*'
        self.abbreviations = {
            'TP.', 'Q.', 'P.', 'St.', 'Dr.', 'Mr.', 'Mrs.', 'Ms.',
            'Prof.', 'vs.', 'etc.', 'Ltd.', 'Inc.', 'Co.',
            'Th.S', 'TS.', 'GS.', 'PGS.'
        }
    
    def split_sentences(self, text: str) -> List[str]:
        """Tách câu bằng regex"""
        # Xử lý trước các viết tắt
        protected_text = text
        for abbr in self.abbreviations:
            protected_text = protected_text.replace(abbr, abbr.replace('.', '<!DOT!>'))
        
        # Tách câu
        sentences = re.split(self.sentence_endings, protected_text)
        
        # Khôi phục dấu chấm trong viết tắt
        sentences = [s.replace('<!DOT!>', '.').strip() for s in sentences if s.strip()]
        
        return sentences

class SimpleVietnameseTokenizer:
    """
    Tokenizer đơn giản cho tiếng Việt - thay thế cho PyVi
    """
    def tokenize(self, text: str) -> str:
        """Tokenize đơn giản bằng cách tách từ"""
        # Xử lý dấu câu
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        # Tách từ
        words = text.split()
        return ' '.join(words)

class VietnameseEmbeddingEvaluator:
    def __init__(self, config_path: str = "config.json"):
        """
        Khởi tạo evaluator với file config
        """
        self.config = self.load_config(config_path)
        self.results = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Khởi tạo Vietnamese NLP tools
        self.sentence_splitter = VietnameseSentenceSplitter()
        self.tokenizer = SimpleVietnameseTokenizer()
        
        logger.info(f"Sử dụng device: {self.device}")
        logger.info(f"Sử dụng Vietnamese sentence splitter: {type(self.sentence_splitter).__name__}")
        
    def load_config(self, config_path: str) -> Dict:
        """Load cấu hình từ file JSON"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Đã load config từ {config_path}")
            return config
        except Exception as e:
            logger.error(f"Lỗi khi load config: {e}")
            raise
    
    def load_document_from_md(self, file_path: str) -> str:
        """Load văn bản từ file markdown"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Đã load document từ {file_path} ({len(content)} ký tự)")
            return content
        except Exception as e:
            logger.error(f"Lỗi khi load document từ {file_path}: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        """
        Chia tài liệu thành chunks với tách câu tiếng Việt
        """
        if chunk_size is None:
            chunk_size = self.config.get("evaluation_settings", {}).get("chunk_size", 200)
        
        logger.info(f"Bắt đầu chia text thành chunks với size={chunk_size}")
        
        try:
            # Tách câu bằng regex splitter
            sentences = self.sentence_splitter.split_sentences(text)
            logger.info(f"Đã tách được {len(sentences)} câu")
            
            chunks = []
            current_chunk = ""
            current_word_count = 0
            
            for i, sentence in enumerate(sentences):
                # Tokenize từ bằng simple tokenizer
                words = self.tokenizer.tokenize(sentence).split()
                
                if current_word_count + len(words) <= chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_word_count += len(words)
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        logger.debug(f"Chunk {len(chunks)}: {len(current_chunk)} chars, {current_word_count} words")
                    
                    current_chunk = sentence
                    current_word_count = len(words)
            
            # Thêm chunk cuối cùng
            if current_chunk:
                chunks.append(current_chunk.strip())
                logger.debug(f"Chunk {len(chunks)}: {len(current_chunk)} chars, {current_word_count} words")
            
            logger.info(f"Đã chia thành {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Lỗi khi chia text: {e}")
            raise
    
    def evaluate_model(self, model_name: str, chunks: List[str]) -> Dict[str, Any]:
        """
        Đánh giá một embedding model với các chunks
        """
        logger.info(f"Bắt đầu đánh giá model: {model_name}")
        
        try:
            # Load model
            logger.info(f"Đang load model {model_name}...")
            model = SentenceTransformer(model_name, device=self.device)
            
            # Thông tin model
            embedding_dim = model.get_sentence_embedding_dimension()
            logger.info(f"Số chiều embedding: {embedding_dim}")
            
            # Tạo embeddings
            logger.info(f"Đang tạo embeddings cho {len(chunks)} chunks...")
            embeddings = model.encode(
                chunks, 
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=32
            )
            
            logger.info(f"Shape của embeddings: {embeddings.shape}")
            
            # Tính similarity matrix
            similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
            
            # Các metrics đánh giá
            metrics = self.calculate_metrics(similarity_matrix, chunks)
            
            result = {
                "model_name": model_name,
                "embedding_dimension": embedding_dim,
                "num_chunks": len(chunks),
                "device_used": str(model.device),
                "similarity_matrix": similarity_matrix,
                "metrics": metrics,
                "status": "success"
            }
            
            logger.info(f"Đánh giá thành công model {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá model {model_name}: {e}")
            return {
                "model_name": model_name,
                "status": "error",
                "error": str(e)
            }
    
    def calculate_metrics(self, similarity_matrix: np.ndarray, chunks: List[str]) -> Dict[str, float]:
        """
        Tính các metrics đánh giá chất lượng embedding
        """
        logger.debug("Đang tính toán metrics...")
        
        # Loại bỏ đường chéo chính (similarity của chunk với chính nó)
        n = similarity_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        similarities = similarity_matrix[mask]
        
        metrics = {
            "avg_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "median_similarity": float(np.median(similarities)),
            "similarity_range": float(np.max(similarities) - np.min(similarities))
        }
        
        # Tính phần trăm similarities trên threshold
        threshold = self.config.get("evaluation_settings", {}).get("similarity_threshold", 0.7)
        high_sim_count = np.sum(similarities > threshold)
        metrics["high_similarity_ratio"] = float(high_sim_count / len(similarities))
        
        logger.debug(f"Metrics calculated: {metrics}")
        return metrics
    
    def evaluate_all_models(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Đánh giá tất cả models trong config
        """
        logger.info("Bắt đầu đánh giá tất cả models...")
        
        results = {}
        models = self.config.get("models", [])
        
        for i, model_config in enumerate(models, 1):
            model_name = model_config["name"]
            logger.info(f"Đánh giá model {i}/{len(models)}: {model_name}")
            
            result = self.evaluate_model(model_name, chunks)
            result["model_description"] = model_config.get("description", "")
            result["language_support"] = model_config.get("language_support", "")
            
            results[model_name] = result
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_dir: str = "reports") -> str:
        """
        Tạo báo cáo so sánh các models
        """
        logger.info("Đang tạo báo cáo...")
        
        # Tạo thư mục output
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{output_dir}/embedding_evaluation_report_{timestamp}.html"
        
        # Chuẩn bị data cho báo cáo
        successful_results = {k: v for k, v in results.items() if v.get("status") == "success"}
        
        if not successful_results:
            logger.error("Không có model nào đánh giá thành công!")
            return None
        
        # Tạo DataFrame cho comparison
        comparison_data = []
        for model_name, result in successful_results.items():
            metrics = result.get("metrics", {})
            row = {
                "Model": model_name,
                "Description": result.get("model_description", ""),
                "Language Support": result.get("language_support", ""),
                "Embedding Dimension": result.get("embedding_dimension", "N/A"),
                "Num Chunks": result.get("num_chunks", "N/A"),
                "Avg Similarity": f"{metrics.get('avg_similarity', 0):.4f}",
                "Std Similarity": f"{metrics.get('std_similarity', 0):.4f}",
                "Similarity Range": f"{metrics.get('similarity_range', 0):.4f}",
                "High Sim Ratio": f"{metrics.get('high_similarity_ratio', 0):.2%}",
                "Status": result.get("status", "unknown")
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Tạo HTML report (tương tự như trước)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vietnamese Embedding Models Evaluation Report</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
                .model-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .success {{ background-color: #f0fff0; }}
                .error {{ background-color: #fff0f0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; }}
                .warning {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🇻🇳 Vietnamese Embedding Models Evaluation Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Total Models Tested:</strong> {len(results)}</p>
                <p><strong>Successful Evaluations:</strong> {len(successful_results)}</p>
            </div>
            
            <div class="warning">
                <strong>⚠️ Note:</strong> Using simplified Vietnamese NLP tools (regex-based) instead of underthesea for Windows compatibility.
                For more accurate results, consider using underthesea via conda-forge.
            </div>
            
            <h2>📊 Models Comparison</h2>
            {df.to_html(index=False, escape=False, classes='comparison-table')}
        </body>
        </html>
        """
        
        # Ghi file
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Đã tạo báo cáo tại: {report_file}")
        return report_file
    
    def run_evaluation(self, document_path: str) -> str:
        """
        Chạy toàn bộ quá trình đánh giá
        """
        logger.info("🚀 Bắt đầu quá trình đánh giá embedding models...")
        
        try:
            # 1. Load document
            document = self.load_document_from_md(document_path)
            
            # 2. Chunk text
            chunks = self.chunk_text(document)
            
            if len(chunks) < 2:
                logger.error("Cần ít nhất 2 chunks để đánh giá!")
                return None
            
            # 3. Evaluate all models
            results = self.evaluate_all_models(chunks)
            
            # 4. Generate report
            report_file = self.generate_report(results)
            
            logger.info("✅ Hoàn thành đánh giá!")
            return report_file
            
        except Exception as e:
            logger.error(f"Lỗi trong quá trình đánh giá: {e}")
            raise

# Hàm main để chạy
def main():
    # Khởi tạo evaluator
    evaluator = VietnameseEmbeddingEvaluator("config.json")
    
    # Chạy đánh giá với document
    document_path = "vietnamese_document.md"
    
    if not os.path.exists(document_path):
        logger.error(f"Không tìm thấy file document: {document_path}")
        logger.info("Vui lòng tạo file vietnamese_document.md với nội dung tiếng Việt để test.")
        return
    
    try:
        report_file = evaluator.run_evaluation(document_path)
        if report_file:
            logger.info(f"📄 Báo cáo đã được tạo tại: {report_file}")
            logger.info("Mở file HTML trong trình duyệt để xem kết quả chi tiết!")
    except Exception as e:
        logger.error(f"Lỗi: {e}")

if __name__ == "__main__":
    main()