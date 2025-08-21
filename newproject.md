Tôi sẽ tạo cho bạn một dự án hoàn chỉnh để đánh giá embedding tiếng Việt dựa trên thiết kế đã cung cấp và tối ưu cho GPU. Dưới đây là cấu trúc dự án chi tiết:

## 1. Cấu trúc thư mục

```
vietnamese_embedding_evaluator/
├── configs/
│   └── models.json
├── data/
│   ├── content.md
│   └── test_suite.json
├── reports/
│   └── # Nơi lưu các file JSON kết quả và biểu đồ
├── src/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── embedding_manager.py
│   ├── metrics.py
│   └── visualizer.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## 2. File requirements.txt

```txt
sentence-transformers>=2.2.2
torch>=2.0.0
torchvision
torchaudio
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
underthesea>=6.7.0
pyvi>=0.1.1
scikit-learn>=1.3.0
transformers>=4.30.0
accelerate>=0.20.0
pathlib
```

## 3. configs/models.json

```json
{
  "models": [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/distiluse-base-multilingual-cased",
    "sentence-transformers/LaBSE",
    "keepitreal/vietnamese-sbert",
    "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
    "vinai/phobert-base-v2"
  ]
}
```

## 4. data/content.md

```markdown
# Lịch sử và Phát triển của Trí tuệ Nhân tạo tại Việt Nam

## Khái niệm về Trí tuệ Nhân tạo

Trí tuệ nhân tạo (AI) là một lĩnh vực của khoa học máy tính tập trung vào việc tạo ra các cỗ máy thông minh có khả năng hoạt động và phản ứng như con người. Thuật ngữ "Artificial Intelligence" được John McCarthy đặt ra lần đầu tiên vào năm 1956 tại Hội nghị Dartmouth.

AI bao gồm nhiều lĩnh vực con như học máy, xử lý ngôn ngữ tự nhiên, thị giác máy tính, và robotics. Mục tiêu cuối cùng của AI là tạo ra những hệ thống có thể thực hiện các tác vụ đòi hỏi trí thông minh của con người.

## Giai đoạn đầu của AI (1950-1970)

Giai đoạn đầu của AI, từ những năm 1950 đến 1970, được gọi là thời kỳ của "AI biểu tượng" hay "GOFAI" (Good Old-Fashioned AI). Các nhà nghiên cứu tin rằng trí thông minh của con người có thể được mô phỏng bằng cách sử dụng logic toán học và các quy tắc biểu tượng rõ ràng.

Trong giai đoạn này, các nhà khoa học tập trung vào việc phát triển các hệ thống chuyên gia và các thuật toán tìm kiếm. Họ tin rằng có thể giải quyết mọi vấn đề bằng cách lập trình các quy tắc logic một cách tường minh.

## Mùa đông AI đầu tiên (1970-1980)

Vào giữa những năm 1970, lĩnh vực AI trải qua "mùa đông AI" đầu tiên do sự cắt giảm tài trợ nghiên cứu và sự thất vọng về tiến độ chậm chạp. Các hệ thống AI thời đó không thể xử lý được sự không chắc chắn và tính mơ hồ của thế giới thực.

Những hạn chế chính bao gồm: khả năng xử lý dữ liệu hạn chế, thiếu sức mạnh tính toán, và việc không thể học hỏi từ kinh nghiệm. Điều này dẫn đến sự giảm sút đáng kể trong đầu tư và nghiên cứu AI.

## Sự trỗi dậy của Mạng nơ-ron (1980-2000)

Sự trỗi dậy của mạng nơ-ron nhân tạo và học máy vào những năm 1980 và 1990 đã mở ra một kỷ nguyên mới cho AI. Thay vì lập trình các quy tắc một cách rõ ràng, các hệ thống bắt đầu có khả năng học hỏi từ dữ liệu.

Các thuật toán như backpropagation được phát triển, cho phép huấn luyện mạng nơ-ron nhiều lớp. Điều này tạo nền tảng cho những đột phá sau này trong lĩnh vực học sâu.

## Kỷ nguyên Học sâu (2010-nay)

Ngày nay, học sâu (Deep Learning), một nhánh của học máy sử dụng mạng nơ-ron sâu, đã tạo ra những đột phá đáng kinh ngạc trong nhiều lĩnh vực. Các ứng dụng bao gồm nhận dạng hình ảnh, xử lý ngôn ngữ tự nhiên, xe tự lái, và y học chính xác.

Các mô hình lớn như GPT-3, GPT-4, BERT, và transformer architecture đã cách mạng hóa cách chúng ta tiếp cận các bài toán AI. Sức mạnh tính toán ngày càng tăng và lượng dữ liệu khổng lồ đã tạo điều kiện cho những tiến bộ này.

## AI tại Việt Nam

Việt Nam đang nhanh chóng phát triển trong lĩnh vực AI với nhiều startup công nghệ và trung tâm nghiên cứu. Các trường đại học hàng đầu như Đại học Bách Khoa Hà Nội, Đại học Quốc Gia TP.HCM đã thành lập các khoa và phòng lab chuyên về AI.

Chính phủ Việt Nam đã ban hành Chiến lược quốc gia về nghiên cứu, phát triển và ứng dụng trí tuệ nhân tạo đến năm 2030. Mục tiêu là biến Việt Nam trở thành một trong những nước dẫn đầu ASEAN về AI.

## Ứng dụng AI trong thực tế

AI đã được ứng dụng rộng rãi trong nhiều lĩnh vực tại Việt Nam như ngân hàng, thương mại điện tử, y tế, giáo dục và nông nghiệp. Các chatbot thông minh, hệ thống gợi ý sản phẩm, và phân tích dữ liệu khách hàng đã trở nên phổ biến.

Trong y tế, AI được sử dụng để chẩn đoán hình ảnh y khoa, dự đoán dịch bệnh, và phát triển thuốc mới. Trong nông nghiệp, AI giúp tối ưu hóa việc tưới tiêu, dự báo thời tiết, và quản lý cây trồng.

## Thách thức và Tương lai

Mặc dù có nhiều tiến bộ, AI vẫn đối mặt với nhiều thách thức như vấn đề đạo đức, bias trong dữ liệu, bảo mật thông tin, và tác động đến việc làm. Việt Nam cần phát triển khung pháp lý phù hợp và đào tạo nhân lực chất lượng cao.

Tương lai của AI tại Việt Nam rất triển vọng với sự đầu tư mạnh mẽ vào nghiên cứu và phát triển. Mục tiêu là tạo ra những sản phẩm AI "Make in Vietnam" có thể cạnh tranh trên thị trường quốc tế.
```

## 5. data/test_suite.json

```json
[
  {
    "question": "Ai là người đầu tiên đặt ra thuật ngữ Trí tuệ nhân tạo?",
    "correct_chunk_id": "chunk_0"
  },
  {
    "question": "Hội nghị Dartmouth diễn ra vào năm nào?",
    "correct_chunk_id": "chunk_0"
  },
  {
    "question": "AI biểu tượng hay GOFAI là gì?",
    "correct_chunk_id": "chunk_1"
  },
  {
    "question": "Giai đoạn đầu của AI tập trung vào những gì?",
    "correct_chunk_id": "chunk_1"
  },
  {
    "question": "Mùa đông AI đầu tiên xảy ra khi nào?",
    "correct_chunk_id": "chunk_2"
  },
  {
    "question": "Nguyên nhân chính gây ra mùa đông AI là gì?",
    "correct_chunk_id": "chunk_2"
  },
  {
    "question": "Thuật toán backpropagation được phát triển vào thời gian nào?",
    "correct_chunk_id": "chunk_3"
  },
  {
    "question": "Mạng nơ-ron nhân tạo trỗi dậy vào giai đoạn nào?",
    "correct_chunk_id": "chunk_3"
  },
  {
    "question": "Học sâu là gì và ứng dụng trong những lĩnh vực nào?",
    "correct_chunk_id": "chunk_4"
  },
  {
    "question": "GPT-4 và BERT thuộc về kỷ nguyên nào của AI?",
    "correct_chunk_id": "chunk_4"
  },
  {
    "question": "Những trường đại học nào ở Việt Nam nghiên cứu về AI?",
    "correct_chunk_id": "chunk_5"
  },
  {
    "question": "Chiến lược quốc gia về AI của Việt Nam có mục tiêu gì?",
    "correct_chunk_id": "chunk_5"
  },
  {
    "question": "AI được ứng dụng trong những lĩnh vực nào tại Việt Nam?",
    "correct_chunk_id": "chunk_6"
  },
  {
    "question": "AI được sử dụng như thế nào trong y tế và nông nghiệp?",
    "correct_chunk_id": "chunk_6"
  },
  {
    "question": "Những thách thức chính của AI hiện nay là gì?",
    "correct_chunk_id": "chunk_7"
  },
  {
    "question": "Tương lai AI tại Việt Nam như thế nào?",
    "correct_chunk_id": "chunk_7"
  }
]
```

## 6. src/data_processor.py

```python
import os
import re
import json
from pathlib import Path
from pyvi import ViTokenizer
import underthesea
from typing import List, Dict

class DataProcessor:
    def __init__(self):
        self.chunk_size = 150  # Số từ tối đa trong một chunk
        self.overlap = 30      # Số từ overlap giữa các chunk
    
    def clean_text(self, text: str) -> str:
        """Làm sạch văn bản tiếng Việt"""
        # Xóa markdown headers
        text = re.sub(r'#+\s*', '', text)
        # Xóa các ký tự đặc biệt không cần thiết
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def segment_vietnamese_text(self, text: str) -> List[str]:
        """Tách từ tiếng Việt"""
        try:
            # Sử dụng underthesea để tách câu
            sentences = underthesea.sent_tokenize(text)
            # Tách từ cho mỗi câu
            segmented_sentences = []
            for sentence in sentences:
                segmented = ViTokenizer.tokenize(sentence)
                segmented_sentences.append(segmented)
            return segmented_sentences
        except Exception as e:
            print(f"Lỗi khi tách từ: {e}")
            return [text]
    
    def create_chunks(self, text: str) -> List[Dict[str, str]]:
        """Chia văn bản thành các chunks với overlap"""
        cleaned_text = self.clean_text(text)
        sentences = self.segment_vietnamese_text(cleaned_text)
        
        chunks = []
        current_chunk_words = []
        chunk_counter = 0
        
        for sentence in sentences:
            words = sentence.split()
            
            # Nếu thêm câu này vào chunk hiện tại vượt quá giới hạn
            if len(current_chunk_words) + len(words) > self.chunk_size:
                if current_chunk_words:
                    # Tạo chunk từ các từ hiện tại
                    chunk_text = ' '.join(current_chunk_words)
                    chunks.append({
                        'id': f'chunk_{chunk_counter}',
                        'text': chunk_text,
                        'word_count': len(current_chunk_words)
                    })
                    chunk_counter += 1
                    
                    # Giữ lại overlap từ chunk trước
                    overlap_words = current_chunk_words[-self.overlap:] if len(current_chunk_words) > self.overlap else current_chunk_words
                    current_chunk_words = overlap_words + words
                else:
                    current_chunk_words = words
            else:
                current_chunk_words.extend(words)
        
        # Thêm chunk cuối cùng nếu có
        if current_chunk_words:
            chunk_text = ' '.join(current_chunk_words)
            chunks.append({
                'id': f'chunk_{chunk_counter}',
                'text': chunk_text,
                'word_count': len(current_chunk_words)
            })
        
        return chunks
    
    def load_and_process_content(self, file_path: str) -> List[Dict[str, str]]:
        """Load và xử lý file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = self.create_chunks(content)
            print(f"Đã tạo {len(chunks)} chunks từ file {file_path}")
            
            # In thống kê
            word_counts = [chunk['word_count'] for chunk in chunks]
            print(f"Số từ trung bình mỗi chunk: {sum(word_counts) / len(word_counts):.1f}")
            print(f"Chunk ngắn nhất: {min(word_counts)} từ")
            print(f"Chunk dài nhất: {max(word_counts)} từ")
            
            return chunks
            
        except FileNotFoundError:
            print(f"Không tìm thấy file: {file_path}")
            return []
        except Exception as e:
            print(f"Lỗi khi xử lý file: {e}")
            return []
    
    def load_test_suite(self, file_path: str) -> List[Dict]:
        """Load bộ test questions"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                test_suite = json.load(f)
            print(f"Đã load {len(test_suite)} câu hỏi test")
            return test_suite
        except Exception as e:
            print(f"Lỗi khi load test suite: {e}")
            return []
    
    def save_chunks_info(self, chunks: List[Dict], output_path: str):
        """Lưu thông tin chunks để debug"""
        chunks_info = {
            'total_chunks': len(chunks),
            'chunks': chunks
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_info, f, ensure_ascii=False, indent=2)
        
        print(f"Đã lưu thông tin chunks vào: {output_path}")
```

## 7. src/embedding_manager.py

```python
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict
import time

class EmbeddingManager:
    def __init__(self, model_name: str, cache_dir: str = "./model_cache"):
        """
        Khởi tạo Embedding Manager với tối ưu GPU
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Kiểm tra GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM khả dụng: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("Sử dụng CPU")
        
        # Load model
        print(f"Đang tải model: {model_name}...")
        start_time = time.time()
        
        try:
            self.model = SentenceTransformer(
                model_name, 
                device=self.device,
                cache_folder=cache_dir
            )
            load_time = time.time() - start_time
            print(f"Đã tải model thành công trong {load_time:.2f}s")
            print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            print(f"Lỗi khi tải model {model_name}: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Lấy thông tin model"""
        return {
            'name': self.model_name,
            'dimension': self.model.get_sentence_embedding_dimension(),
            'device': self.device,
            'max_sequence_length': getattr(self.model, 'max_seq_length', 'Unknown')
        }
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> torch.Tensor:
        """
        Encode danh sách texts thành embeddings với tối ưu GPU
        """
        if not texts:
            return torch.empty(0, self.model.get_sentence_embedding_dimension())
        
        print(f"Đang encode {len(texts)} texts...")
        start_time = time.time()
        
        try:
            # Tối ưu batch size dựa trên VRAM
            if self.device == "cuda":
                available_memory = torch.cuda.get_device_properties(0).total_memory
                if available_memory < 4e9:  # < 4GB
                    batch_size = min(16, batch_size)
                elif available_memory < 8e9:  # < 8GB  
                    batch_size = min(32, batch_size)
                else:  # >= 8GB
                    batch_size = min(64, batch_size)
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True  # Normalize để tối ưu cosine similarity
            )
            
            encode_time = time.time() - start_time
            print(f"Hoàn thành encoding trong {encode_time:.2f}s")
            print(f"Shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            print(f"Lỗi khi encode: {e}")
            raise
    
    def find_most_similar(self, query_embedding: torch.Tensor, 
                         corpus_embeddings: torch.Tensor, 
                         top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tìm top-k embeddings tương đồng nhất với query
        """
        try:
            # Tính cosine similarity
            similarities = util.cos_sim(query_embedding, corpus_embeddings)
            
            # Lấy top-k results
            top_k = min(top_k, corpus_embeddings.shape[0])
            top_results = torch.topk(similarities, k=top_k, dim=-1)
            
            indices = top_results.indices.cpu().numpy().flatten()
            scores = top_results.values.cpu().numpy().flatten()
            
            return indices, scores
            
        except Exception as e:
            print(f"Lỗi khi tìm kiếm tương đồng: {e}")
            raise
    
    def batch_search(self, queries: List[str], corpus_embeddings: torch.Tensor, 
                    chunk_ids: List[str], top_k: int = 5) -> List[Dict]:
        """
        Thực hiện batch search cho nhiều queries
        """
        print(f"Đang thực hiện batch search cho {len(queries)} queries...")
        
        # Encode tất cả queries
        query_embeddings = self.encode_texts(queries, show_progress=False)
        
        results = []
        for i, (query, query_emb) in enumerate(zip(queries, query_embeddings)):
            indices, scores = self.find_most_similar(
                query_emb.unsqueeze(0), 
                corpus_embeddings, 
                top_k
            )
            
            top_chunks = []
            for idx, score in zip(indices, scores):
                top_chunks.append({
                    'chunk_id': chunk_ids[idx],
                    'score': float(score),
                    'rank': len(top_chunks) + 1
                })
            
            results.append({
                'query': query,
                'top_results': top_chunks
            })
        
        return results
    
    def clear_cache(self):
        """Dọn dẹp cache GPU"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("Đã dọn dẹp GPU cache")
    
    def get_memory_usage(self) -> Dict:
        """Lấy thông tin sử dụng memory"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            return {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'device': torch.cuda.get_device_name(0)
            }
        else:
            return {'device': 'CPU', 'allocated_gb': 0, 'cached_gb': 0}
```

## 8. src/metrics.py

```python
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict

class MetricsCalculator:
    def __init__(self):
        self.metrics_names = [
            'MRR', 'Hit_Rate@1', 'Hit_Rate@3', 'Hit_Rate@5', 
            'MAP@5', 'NDCG@5', 'Precision@5', 'Recall@5'
        ]
    
    def calculate_reciprocal_rank(self, correct_chunk_id: str, ranked_results: List[Dict]) -> float:
        """Tính Reciprocal Rank cho một query"""
        for rank, result in enumerate(ranked_results, 1):
            if result['chunk_id'] == correct_chunk_id:
                return 1.0 / rank
        return 0.0
    
    def calculate_hit_rate_at_k(self, correct_chunk_id: str, ranked_results: List[Dict], k: int) -> int:
        """Tính Hit Rate@k cho một query"""
        top_k_ids = [result['chunk_id'] for result in ranked_results[:k]]
        return 1 if correct_chunk_id in top_k_ids else 0
    
    def calculate_precision_at_k(self, correct_chunk_id: str, ranked_results: List[Dict], k: int) -> float:
        """Tính Precision@k"""
        top_k_ids = [result['chunk_id'] for result in ranked_results[:k]]
        relevant_found = sum(1 for chunk_id in top_k_ids if chunk_id == correct_chunk_id)
        return relevant_found / min(k, len(ranked_results))
    
    def calculate_recall_at_k(self, correct_chunk_id: str, ranked_results: List[Dict], k: int) -> float:
        """Tính Recall@k (trong trường hợp này, mỗi query chỉ có 1 correct answer)"""
        top_k_ids = [result['chunk_id'] for result in ranked_results[:k]]
        return 1.0 if correct_chunk_id in top_k_ids else 0.0
    
    def calculate_average_precision(self, correct_chunk_id: str, ranked_results: List[Dict]) -> float:
        """Tính Average Precision"""
        precision_at_k = []
        for k in range(1, len(ranked_results) + 1):
            if ranked_results[k-1]['chunk_id'] == correct_chunk_id:
                precision = self.calculate_precision_at_k(correct_chunk_id, ranked_results, k)
                precision_at_k.append(precision)
        
        return np.mean(precision_at_k) if precision_at_k else 0.0
    
    def calculate_ndcg_at_k(self, correct_chunk_id: str, ranked_results: List[Dict], k: int) -> float:
        """Tính NDCG@k"""
        # Tính DCG@k
        dcg = 0.0
        for i, result in enumerate(ranked_results[:k]):
            if result['chunk_id'] == correct_chunk_id:
                relevance = 1  # Binary relevance
                dcg += relevance / np.log2(i + 2)  # i+2 vì log2(1) = 0
        
        # Tính IDCG@k (trong trường hợp này là 1.0 vì chỉ có 1 correct answer)
        idcg = 1.0  # 1 / log2(2) = 1
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_all_metrics(self, test_results: List[Dict]) -> Dict:
        """Tính tất cả metrics cho toàn bộ test results"""
        if not test_results:
            return {metric: 0.0 for metric in self.metrics_names}
        
        rr_scores = []
        hit_1_scores = []
        hit_3_scores = []
        hit_5_scores = []
        ap_scores = []
        ndcg_5_scores = []
        precision_5_scores = []
        recall_5_scores = []
        
        for result in test_results:
            correct_id = result['correct_chunk_id']
            ranked_results = result['top_5_results']
            
            # Reciprocal Rank
            rr = self.calculate_reciprocal_rank(correct_id, ranked_results)
            rr_scores.append(rr)
            
            # Hit Rates
            hit_1_scores.append(self.calculate_hit_rate_at_k(correct_id, ranked_results, 1))
            hit_3_scores.append(self.calculate_hit_rate_at_k(correct_id, ranked_results, 3))
            hit_5_scores.append(self.calculate_hit_rate_at_k(correct_id, ranked_results, 5))
            
            # Average Precision
            ap = self.calculate_average_precision(correct_id, ranked_results)
            ap_scores.append(ap)
            
            # NDCG@5
            ndcg_5 = self.calculate_ndcg_at_k(correct_id, ranked_results, 5)
            ndcg_5_scores.append(ndcg_5)
            
            # Precision@5 và Recall@5
            precision_5 = self.calculate_precision_at_k(correct_id, ranked_results, 5)
            recall_5 = self.calculate_recall_at_k(correct_id, ranked_results, 5)
            precision_5_scores.append(precision_5)
            recall_5_scores.append(recall_5)
        
        # Tính mean cho tất cả metrics
        metrics = {
            'MRR': np.mean(rr_scores),
            'Hit_Rate@1': np.mean(hit_1_scores),
            'Hit_Rate@3': np.mean(hit_3_scores),
            'Hit_Rate@5': np.mean(hit_5_scores),
            'MAP@5': np.mean(ap_scores),
            'NDCG@5': np.mean(ndcg_5_scores),
            '
```python
            'Precision@5': np.mean(precision_5_scores),
            'Recall@5': np.mean(recall_5_scores)
        }
        
        return metrics
    
    def calculate_detailed_metrics(self, test_results: List[Dict]) -> Tuple[Dict, pd.DataFrame]:
        """Tính metrics chi tiết cho từng câu hỏi"""
        detailed_results = []
        
        for i, result in enumerate(test_results):
            correct_id = result['correct_chunk_id']
            ranked_results = result['top_5_results']
            question = result['question']
            
            # Tìm rank của correct answer
            found_rank = None
            for rank, res in enumerate(ranked_results, 1):
                if res['chunk_id'] == correct_id:
                    found_rank = rank
                    break
            
            detailed_result = {
                'question_id': i + 1,
                'question': question[:50] + "..." if len(question) > 50 else question,
                'correct_chunk_id': correct_id,
                'found_rank': found_rank,
                'reciprocal_rank': self.calculate_reciprocal_rank(correct_id, ranked_results),
                'hit@1': self.calculate_hit_rate_at_k(correct_id, ranked_results, 1),
                'hit@3': self.calculate_hit_rate_at_k(correct_id, ranked_results, 3),
                'hit@5': self.calculate_hit_rate_at_k(correct_id, ranked_results, 5),
                'ap': self.calculate_average_precision(correct_id, ranked_results),
                'ndcg@5': self.calculate_ndcg_at_k(correct_id, ranked_results, 5),
                'top_1_score': ranked_results[0]['score'] if ranked_results else 0.0,
                'correct_score': next((res['score'] for res in ranked_results if res['chunk_id'] == correct_id), 0.0)
            }
            detailed_results.append(detailed_result)
        
        df_detailed = pd.DataFrame(detailed_results)
        summary_metrics = self.calculate_all_metrics(test_results)
        
        return summary_metrics, df_detailed
    
    def get_performance_analysis(self, df_detailed: pd.DataFrame) -> Dict:
        """Phân tích hiệu suất chi tiết"""
        analysis = {
            'total_questions': len(df_detailed),
            'questions_answered_correctly_at_rank_1': int(df_detailed['hit@1'].sum()),
            'questions_answered_correctly_at_rank_3': int(df_detailed['hit@3'].sum()),
            'questions_answered_correctly_at_rank_5': int(df_detailed['hit@5'].sum()),
            'questions_not_found_in_top_5': int((df_detailed['found_rank'].isna()).sum()),
            'average_correct_answer_rank': df_detailed[df_detailed['found_rank'].notna()]['found_rank'].mean(),
            'median_correct_answer_rank': df_detailed[df_detailed['found_rank'].notna()]['found_rank'].median(),
            'score_statistics': {
                'top_1_score_mean': df_detailed['top_1_score'].mean(),
                'top_1_score_std': df_detailed['top_1_score'].std(),
                'correct_score_mean': df_detailed[df_detailed['correct_score'] > 0]['correct_score'].mean(),
                'correct_score_std': df_detailed[df_detailed['correct_score'] > 0]['correct_score'].std()
            }
        }
        
        return analysis
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """So sánh hiệu suất giữa các models"""
        comparison_data = []
        
        for model_name, metrics in model_results.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Sắp xếp theo MRR (cao nhất trước)
        df_comparison = df_comparison.sort_values('MRR', ascending=False).reset_index(drop=True)
        
        # Thêm ranking cho từng metric
        for metric in self.metrics_names:
            if metric in df_comparison.columns:
                df_comparison[f'{metric}_rank'] = df_comparison[metric].rank(ascending=False, method='min')
        
        return df_comparison
```

## 9. src/visualizer.py

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

# Cấu hình matplotlib cho tiếng Việt
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'DejaVu Sans']
sns.set_style("whitegrid")
sns.set_palette("husl")

class ResultVisualizer:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Cấu hình màu sắc
        self.colors = sns.color_palette("husl", 10)
        
    def plot_model_comparison(self, df_comparison: pd.DataFrame, save_path: str = None) -> None:
        """Vẽ biểu đồ so sánh hiệu suất các models"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('So sánh Hiệu suất các Models Embedding Tiếng Việt', fontsize=16, fontweight='bold')
        
        # MRR Comparison
        axes[0, 0].barh(df_comparison['Model'], df_comparison['MRR'], color=self.colors[0])
        axes[0, 0].set_title('Mean Reciprocal Rank (MRR)')
        axes[0, 0].set_xlabel('MRR Score')
        for i, v in enumerate(df_comparison['MRR']):
            axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Hit Rates Comparison
        hit_metrics = ['Hit_Rate@1', 'Hit_Rate@3', 'Hit_Rate@5']
        x = np.arange(len(df_comparison['Model']))
        width = 0.25
        
        for i, metric in enumerate(hit_metrics):
            if metric in df_comparison.columns:
                axes[0, 1].bar(x + i*width, df_comparison[metric], width, 
                              label=metric, color=self.colors[i+1])
        
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Hit Rate')
        axes[0, 1].set_title('Hit Rates Comparison')
        axes[0, 1].set_xticks(x + width)
        axes[0, 1].set_xticklabels([name.split('/')[-1] for name in df_comparison['Model']], rotation=45)
        axes[0, 1].legend()
        
        # NDCG@5 và MAP@5
        if 'NDCG@5' in df_comparison.columns and 'MAP@5' in df_comparison.columns:
            x = np.arange(len(df_comparison['Model']))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, df_comparison['NDCG@5'], width, 
                          label='NDCG@5', color=self.colors[4])
            axes[1, 0].bar(x + width/2, df_comparison['MAP@5'], width, 
                          label='MAP@5', color=self.colors[5])
            
            axes[1, 0].set_xlabel('Models')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_title('NDCG@5 và MAP@5 Comparison')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([name.split('/')[-1] for name in df_comparison['Model']], rotation=45)
            axes[1, 0].legend()
        
        # Overall Ranking
        ranking_metrics = ['MRR_rank', 'Hit_Rate@1_rank', 'Hit_Rate@3_rank', 'Hit_Rate@5_rank']
        available_rankings = [col for col in ranking_metrics if col in df_comparison.columns]
        
        if available_rankings:
            avg_rank = df_comparison[available_rankings].mean(axis=1)
            axes[1, 1].barh(df_comparison['Model'], avg_rank, color=self.colors[6])
            axes[1, 1].set_title('Average Ranking (Lower is Better)')
            axes[1, 1].set_xlabel('Average Rank')
            for i, v in enumerate(avg_rank):
                axes[1, 1].text(v + 0.1, i, f'{v:.1f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu biểu đồ so sánh models: {save_path}")
        
        plt.show()
    
    def plot_detailed_analysis(self, model_name: str, df_detailed: pd.DataFrame, 
                              metrics_summary: Dict, save_path: str = None) -> None:
        """Vẽ biểu đồ phân tích chi tiết cho một model"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        model_short_name = model_name.split('/')[-1]
        fig.suptitle(f'Phân tích Chi tiết - {model_short_name}', fontsize=16, fontweight='bold')
        
        # 1. Distribution of Found Ranks
        ranks = df_detailed[df_detailed['found_rank'].notna()]['found_rank']
        if len(ranks) > 0:
            axes[0, 0].hist(ranks, bins=range(1, 7), alpha=0.7, color=self.colors[0])
            axes[0, 0].set_xlabel('Rank của Câu trả lời Đúng')
            axes[0, 0].set_ylabel('Số lượng Câu hỏi')
            axes[0, 0].set_title('Phân bố Rank của Câu trả lời Đúng')
            axes[0, 0].set_xticks(range(1, 6))
        
        # 2. Score Distribution
        correct_scores = df_detailed[df_detailed['correct_score'] > 0]['correct_score']
        top1_scores = df_detailed['top_1_score']
        
        axes[0, 1].hist([correct_scores, top1_scores], label=['Correct Answer Score', 'Top-1 Score'], 
                       alpha=0.7, color=self.colors[1:3])
        axes[0, 1].set_xlabel('Cosine Similarity Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Phân bố Điểm số')
        axes[0, 1].legend()
        
        # 3. Performance by Question Position
        question_performance = df_detailed.groupby(df_detailed.index // 4)['hit@1'].mean()  # Group every 4 questions
        axes[0, 2].plot(question_performance.index, question_performance.values, 
                       marker='o', color=self.colors[3])
        axes[0, 2].set_xlabel('Nhóm Câu hỏi (mỗi nhóm 4 câu)')
        axes[0, 2].set_ylabel('Hit Rate@1')
        axes[0, 2].set_title('Hiệu suất theo Nhóm Câu hỏi')
        
        # 4. Metrics Overview
        metrics_names = ['MRR', 'Hit_Rate@1', 'Hit_Rate@3', 'Hit_Rate@5', 'MAP@5', 'NDCG@5']
        available_metrics = [m for m in metrics_names if m in metrics_summary]
        metric_values = [metrics_summary[m] for m in available_metrics]
        
        axes[1, 0].bar(available_metrics, metric_values, color=self.colors[4])
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Tổng quan Các Chỉ số')
        axes[1, 0].set_xticklabels(available_metrics, rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(metric_values):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 5. Question Difficulty Analysis
        df_detailed['difficulty'] = 'Easy'
        df_detailed.loc[df_detailed['found_rank'] > 3, 'difficulty'] = 'Hard'
        df_detailed.loc[df_detailed['found_rank'].isna(), 'difficulty'] = 'Very Hard'
        
        difficulty_counts = df_detailed['difficulty'].value_counts()
        axes[1, 1].pie(difficulty_counts.values, labels=difficulty_counts.index, 
                      autopct='%1.1f%%', colors=self.colors[5:8])
        axes[1, 1].set_title('Phân loại Độ khó Câu hỏi')
        
        # 6. Score Gap Analysis
        df_detailed['score_gap'] = df_detailed['top_1_score'] - df_detailed['correct_score']
        score_gaps = df_detailed[df_detailed['correct_score'] > 0]['score_gap']
        
        axes[1, 2].hist(score_gaps, bins=20, alpha=0.7, color=self.colors[8])
        axes[1, 2].set_xlabel('Score Gap (Top-1 - Correct)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Khoảng cách Điểm số')
        axes[1, 2].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu biểu đồ phân tích chi tiết: {save_path}")
        
        plt.show()
    
    def create_performance_heatmap(self, all_model_results: Dict, save_path: str = None) -> None:
        """Tạo heatmap so sánh hiệu suất tất cả models"""
        # Chuẩn bị dữ liệu
        models = []
        metrics_data = []
        
        for model_name, result in all_model_results.items():
            models.append(model_name.split('/')[-1])
            metrics = result['summary_metrics']
            metrics_data.append([
                metrics.get('MRR', 0),
                metrics.get('Hit_Rate@1', 0),
                metrics.get('Hit_Rate@3', 0),
                metrics.get('Hit_Rate@5', 0),
                metrics.get('MAP@5', 0),
                metrics.get('NDCG@5', 0)
            ])
        
        df_heatmap = pd.DataFrame(
            metrics_data, 
            index=models,
            columns=['MRR', 'Hit@1', 'Hit@3', 'Hit@5', 'MAP@5', 'NDCG@5']
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_heatmap, annot=True, cmap='YlOrRd', fmt='.3f', 
                   cbar_kws={'label': 'Performance Score'})
        plt.title('Performance Heatmap - All Models', fontsize=14, fontweight='bold')
        plt.ylabel('Models')
        plt.xlabel('Metrics')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Đã lưu heatmap: {save_path}")
        
        plt.show()
    
    def generate_performance_report(self, all_model_results: Dict, 
                                   output_file: str = "performance_report.html") -> None:
        """Tạo báo cáo HTML tổng hợp"""
        html_content = """
        <html>
        <head>
            <title>Vietnamese Embedding Models - Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
                th { background-color: #f8f9fa; font-weight: bold; }
                .best { background-color: #d4edda; font-weight: bold; }
                .second { background-color: #e2f3ff; }
                .summary { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
        """
        
        html_content += "<h1>Vietnamese Embedding Models - Performance Report</h1>"
        
        # Tạo bảng so sánh
        models_data = []
        for model_name, result in all_model_results.items():
            models_data.append({
                'Model': model_name.split('/')[-1],
                'Full_Name': model_name,
                'Dimension': result.get('embedding_dimension', 'N/A'),
                'Time': result.get('evaluation_time_seconds', 'N/A'),
                **result['summary_metrics']
            })
        
        df = pd.DataFrame(models_data)
        df = df.sort_values('MRR', ascending=False)
        
        html_content += "<h2>Model Performance Comparison</h2>"
        html_content += df.to_html(classes='performance-table', escape=False, index=False)
        
        # Top performer summary
        best_model = df.iloc[0]
        html_content += f"""
        <div class="summary">
            <h3>🏆 Best Performing Model</h3>
            <p><strong>{best_model['Model']}</strong> achieved the highest MRR score of <strong>{best_model['MRR']:.4f}</strong></p>
            <ul>
                <li>Hit Rate@1: {best_model['Hit_Rate@1']:.2%}</li>
                <li>Hit Rate@5: {best_model['Hit_Rate@5']:.2%}</li>
                <li>Embedding Dimension: {best_model['Dimension']}</li>
                <li>Evaluation Time: {best_model['Time']:.2f}s</li>
            </ul>
        </div>
        """
        
        html_content += "</body></html>"
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Đã tạo báo cáo HTML: {output_path}")
```

## 10. evaluate.py (Script chính)

```python
import os
import json
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import numpy as np
from pyvi import ViTokenizer
import underthesea
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import torch

from src.data_processor import DataProcessor
from src.embedding_manager import EmbeddingManager
from src.metrics import MetricsCalculator
from src.visualizer import ResultVisualizer

def setup_directories():
    """Tạo các thư mục cần thiết"""
    directories = ['reports', 'reports/visualizations', 'model_cache', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def load_configuration():
    """Load cấu hình models và test suite"""
    print("🔧 Đang load cấu hình...")
    
    # Load models list
    try:
        with open('configs/models.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        models_to_test = config['models']
        print(f"✅ Đã load {len(models_to_test)} models để test")
    except Exception as e:
        print(f"❌ Lỗi khi load models config: {e}")
        return None, None
    
    # Load test suite
    try:
        with open('data/test_suite.json', 'r', encoding='utf-8') as f:
            test_suite = json.load(f)
        print(f"✅ Đã load {len(test_suite)} câu hỏi test")
    except Exception as e:
        print(f"❌ Lỗi khi load test suite: {e}")
        return None, None
    
    return models_to_test, test_suite

def evaluate_single_model(model_name: str, chunks: list, test_suite: list, 
                         chunk_texts: list, chunk_ids: list) -> dict:
    """Đánh giá một model duy nhất"""
    print(f"\n🚀 Bắt đầu đánh giá model: {model_name}")
    start_time = time.time()
    
    try:
        # Khởi tạo embedding manager
        manager = EmbeddingManager(model_name)
        model_info = manager.get_model_info()
        
        # Encode corpus
        print("📝 Đang encode corpus...")
        corpus_embeddings = manager.encode_texts(chunk_texts)
        
        # Process test cases
        print("🔍 Đang thực hiện tìm kiếm cho các câu hỏi test...")
        test_case_results = []
        
        for i, test_case in enumerate(test_suite):
            question = test_case['question']
            correct_chunk_id = test_case['correct_chunk_id']
            
            # Encode question
            query_embedding = manager.encode_texts([question], show_progress=False)
            
            # Find similar chunks
            top_indices, top_scores = manager.find_most_similar(
                query_embedding, corpus_embeddings, top_k=5
            )
            
            # Build results
            top_results = []
            for idx, score in zip(top_indices, top_scores):
                top_results.append({
                    'chunk_id': chunk_ids[idx],
                    'score': float(score),
                    'text_preview': chunks[idx]['text'][:100] + "..."
                })
            
            # Find rank of correct answer
            found_rank = None
            for rank, result in enumerate(top_results, 1):
                if result['chunk_id'] == correct_chunk_id:
                    found_rank = rank
                    break
            
            test_case_results.append({
                'question': question,
                'correct_chunk_id': correct_chunk_id,
                'found_rank': found_rank,
                'top_5_results': top_results
            })
            
            if (i + 1) % 5 == 0:
                print(f"  Đã xử lý {i + 1}/{len(test_suite)} câu hỏi")
        
        # Calculate metrics
        print("📊 Đang tính toán metrics...")
        calculator = MetricsCalculator()
        summary_metrics, detailed_df = calculator.calculate_detailed_metrics(test_case_results)
        performance_analysis = calculator.get_performance_analysis(detailed_df)
        
        # Memory info
        memory_info = manager.get_memory_usage()
        
        # Clean up
        manager.clear_cache()
        del manager
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        evaluation_time = time.time() - start_time
        print(f"✅ Hoàn thành đánh giá {model_name} trong {evaluation_time:.2f}s")
        print(f"   MRR: {summary_metrics['MRR']:.4f}")
        print(f"   Hit Rate@1: {summary_metrics['Hit_Rate@1']:.2%}")
        
        return {
            'model_name': model_name,
            'model_info': model_info,
            'embedding_dimension': model_info['dimension'],
            'evaluation_time_seconds': evaluation_time,
            'summary_metrics': summary_metrics,
            'detailed_metrics': detailed_df.to_dict('records'),
            'performance_analysis': performance_analysis,
            'test_cases': test_case_results,
            'memory_usage': memory_info,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Lỗi khi đánh giá model {model_name}: {e}")
        return None

def save_model_report(model_result: dict, reports_dir: Path):
    """Lưu báo cáo cho một model"""
    if not model_result:
        return
    
    model_name = model_result['model_name']
    safe_name = model_name.replace('/', '_').replace(':', '_')
    
    # Save detailed JSON report
    json_path = reports_dir / f"{safe_name}_detailed.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(model_result, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Đã lưu báo cáo chi tiết: {json_path}")
    
    # Save summary CSV
    summary_data = {
        'Model': model_name,
        'Dimension': model_result['embedding_dimension'],
        'Time(s)': model_result['evaluation_time_seconds'],
        **model_result['summary_metrics']
    }
    
    csv_path = reports_dir / f"{safe_name}_summary.csv"
    pd.DataFrame([summary_data]).to_csv(csv_path, index=False)
    
    return json_path

def create_visualizations(all_results: dict, visualizer: ResultVisualizer):
    """Tạo tất cả các visualizations"""
    print("\n📈 Đang tạo visualizations...")
    
    try:
        # Model comparison chart
        calculator = MetricsCalculator()
        model_metrics = {name: result['summary_metrics'] 
                        for name, result in all_results.items() 
                        if result is not None}
        
        if len(model_metrics) > 1:
            df_comparison = calculator.compare_models(model_metrics)
            
            # Save comparison plot
            comparison_path = visualizer.output_dir / "visualizations" / "model_comparison.png"
            visualizer.plot_model_comparison(df_comparison, str(comparison_path))
            
            # Create performance heatmap
            heatmap_path = visualizer.output_dir / "visualizations" / "performance_heatmap.png"
            visualizer.create_performance_heatmap(all_results, str(heatmap_path))
            
            # Save comparison data
            comparison_csv = visualizer.output_dir / "model_comparison.csv"
            df_comparison.to_csv(comparison_csv, index=False)
            print(f"💾 Đã lưu bảng so sánh: {comparison_csv}")
        
        # Individual model analysis
        for model_name, result in all_results.items():
            if result is not None:
                detailed_df = pd.DataFrame(result['detailed_metrics'])
                safe_name = model_name.replace('/', '_').replace(':', '_')
                
                detail_path = visualizer.output_dir / "visualizations" / f"{safe_name}_analysis.png"
                visualizer.plot_detailed_analysis(
                    model_name, detailed_df, result['summary_metrics'], str(detail_path)
                )
        
        # Generate HTML report
        visualizer.generate_performance_report(all_results)
        
        print("✅ Đã tạo xong tất cả visualizations")
        
    except Exception as e:
        print(f"❌ Lỗi khi tạo visualizations: {e}")

def main():
    """Main evaluation pipeline"""
    print("=" * 70)
    print("🇻🇳 VIETNAMESE EMBEDDING MODELS EVALUATION PIPELINE")
    print("=" * 70)
    
    # Setup
    setup_directories()
    models_to_test, test_suite = load_configuration()
    
    if not models_to_test or not test_suite:
        print("❌ Không thể load cấu hình. Dừng chương trình.")
        return
    
    # Process data
    print("\n📚 Đang xử lý dữ liệu...")
    processor = DataProcessor()
    chunks = processor.load_and_process_content('data/content.md')
    
    if not chunks:
        print("❌ Không thể load content data. Dừng chương trình.")
        return
    
    chunk_texts = [chunk['text'] for chunk in chunks]
    chunk_ids = [chunk['id'] for chunk in chunks]
    
    # Save chunks info for debugging
    chunks_info_path = Path("reports") / "chunks_info.json"
    processor.save_chunks_info(chunks, chunks_info_path)
    
    # Initialize components
    reports_dir = Path("reports")
    visualizer = ResultVisualizer(str(reports_dir))
    all_results = {}
    
    # Evaluate each model
    print(f"\n🔄 Bắt đầu đánh giá {len(models_to_test)} models...")
    
    for i, model_name in enumerate(models_to_test, 1):
        print(f"\n{'='*50}")
        print(f"📊 Model {i}/{len(models_to_test)}: {model_name}")
        print(f"{'='*50}")
        
        try:
            # Evaluate model
            result = evaluate_single_model(
                model_name, chunks, test_suite, chunk_texts, chunk_ids
            )
            
            if result:
                all_results[model_name] = result
                save_model_report(result, reports_dir)
            
        except KeyboardInterrupt:
            print("\n⚠️  Nhận được tín hiệu dừng. Đang lưu kết quả hiện tại
```python
            print("\n⚠️  Nhận được tín hiệu dừng. Đang lưu kết quả hiện tại...")
            break
        except Exception as e:
            print(f"❌ Lỗi không mong muốn với model {model_name}: {e}")
            continue
    
    # Create visualizations and final reports
    if all_results:
        create_visualizations(all_results, visualizer)
        
        # Save consolidated results
        consolidated_path = reports_dir / "all_results_consolidated.json"
        with open(consolidated_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Đã lưu kết quả tổng hợp: {consolidated_path}")
        
        # Print final summary
        print("\n" + "="*70)
        print("📋 TỔNG KẾT KỀT QUẢ ĐÁNH GIÁ")
        print("="*70)
        
        # Sort by MRR
        sorted_results = sorted(
            [(name, result['summary_metrics']['MRR']) for name, result in all_results.items()],
            key=lambda x: x[1], reverse=True
        )
        
        print("🏆 BẢNG XẾP HẠNG (theo MRR):")
        for rank, (model_name, mrr) in enumerate(sorted_results, 1):
            result = all_results[model_name]
            hit_1 = result['summary_metrics']['Hit_Rate@1']
            time_taken = result['evaluation_time_seconds']
            
            print(f"{rank:2d}. {model_name.split('/')[-1]:<40} "
                  f"MRR: {mrr:.4f} | Hit@1: {hit_1:.2%} | Time: {time_taken:.1f}s")
        
        best_model = sorted_results[0][0]
        best_metrics = all_results[best_model]['summary_metrics']
        
        print(f"\n🥇 CHAMPION: {best_model}")
        print(f"   📊 MRR: {best_metrics['MRR']:.4f}")
        print(f"   🎯 Hit Rate@1: {best_metrics['Hit_Rate@1']:.2%}")
        print(f"   📈 Hit Rate@5: {best_metrics['Hit_Rate@5']:.2%}")
        
        print(f"\n📁 Tất cả báo cáo đã được lưu trong thư mục: {reports_dir.absolute()}")
        print("📊 Kiểm tra file 'performance_report.html' để xem báo cáo chi tiết")
        
    else:
        print("❌ Không có kết quả nào được tạo ra. Vui lòng kiểm tra cấu hình và dữ liệu.")
    
    print("\n🎉 Hoàn thành pipeline đánh giá!")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Chương trình đã bị dừng bởi người dùng.")
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {e}")
        import traceback
        traceback.print_exc()
```

## 11. README.md

```markdown
# Vietnamese Embedding Models Evaluation Pipeline 🇻🇳

Một pipeline tự động để đánh giá và so sánh hiệu suất của các model embedding mã nguồn mở trên các tác vụ tìm kiếm ngữ nghĩa (semantic search) cho tiếng Việt.

## ✨ Tính năng

- **Đánh giá tự động**: So sánh nhiều models embedding cùng lúc
- **Metrics toàn diện**: MRR, Hit Rate@k, MAP, NDCG, Precision, Recall
- **Tối ưu GPU**: Hỗ trợ CUDA với quản lý memory thông minh
- **Visualization**: Biểu đồ và báo cáo HTML chi tiết
- **Tiếng Việt**: Tối ưu cho văn bản tiếng Việt với pyvi và underthesea

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone <your-repo>
cd vietnamese_embedding_evaluator
```

### 2. Tạo virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Cài đặt thêm cho GPU (nếu có)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Cách sử dụng

### Chạy evaluation pipeline
```bash
python evaluate.py
```

### Cấu trúc thư mục sau khi chạy
```
vietnamese_embedding_evaluator/
├── reports/
│   ├── model_comparison.csv              # Bảng so sánh các models
│   ├── performance_report.html          # Báo cáo HTML chi tiết
│   ├── all_results_consolidated.json    # Kết quả tổng hợp
│   ├── chunks_info.json                 # Thông tin các text chunks
│   ├── visualizations/                  # Thư mục chứa các biểu đồ
│   └── *_detailed.json                  # Báo cáo chi tiết từng model
└── model_cache/                         # Cache các models đã tải
```

## 🎯 Metrics được đánh giá

- **MRR (Mean Reciprocal Rank)**: Chỉ số chính để đánh giá chất lượng ranking
- **Hit Rate@k**: Tỷ lệ câu hỏi có đáp án đúng trong top-k (k=1,3,5)
- **MAP@5**: Mean Average Precision tại top-5
- **NDCG@5**: Normalized Discounted Cumulative Gain tại top-5
- **Precision@5 & Recall@5**: Độ chính xác và độ bao phủ

## 🔧 Tùy chỉnh

### Thêm models mới
Chỉnh sửa `configs/models.json`:
```json
{
  "models": [
    "your-new-model/from-huggingface",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  ]
}
```

### Thay đổi dữ liệu test
- Chỉnh sửa `data/content.md` với nội dung mới
- Cập nhật `data/test_suite.json` với câu hỏi tương ứng

### Điều chỉnh chunk size
Trong `src/data_processor.py`, thay đổi:
```python
self.chunk_size = 150  # Số từ tối đa trong một chunk
self.overlap = 30      # Số từ overlap giữa các chunk
```

## 📈 Kết quả mẫu

```
🏆 BẢNG XẾP HẠNG (theo MRR):
 1. paraphrase-multilingual-mpnet-base-v2     MRR: 0.7234 | Hit@1: 62.50% | Time: 45.2s
 2. LaBSE                                     MRR: 0.6891 | Hit@1: 56.25% | Time: 52.1s
 3. vietnamese-sbert                          MRR: 0.6456 | Hit@1: 50.00% | Time: 38.7s
```

## 🛠️ Troubleshooting

### Lỗi CUDA out of memory
- Giảm batch_size trong `embedding_manager.py`
- Đánh giá từng model một: comment bớt models trong `configs/models.json`

### Lỗi tách từ tiếng Việt
```bash
# Cài đặt lại underthesea
pip uninstall underthesea
pip install underthesea
```

### Model không tải được
- Kiểm tra kết nối internet
- Xóa `model_cache/` và chạy lại

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push: `git push origin feature/new-feature`
5. Tạo Pull Request

## 📝 License

MIT License - xem file LICENSE để biết chi tiết.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/)
- [underthesea](https://github.com/undertheseanlp/underthesea)
- [pyvi](https://github.com/trungtv/pyvi)
- Cộng đồng NLP Việt Nam
```

## 12. Cách chạy dự án

1. **Tạo cấu trúc thư mục** theo đúng layout trên
2. **Cài đặt dependencies**:
```bash
pip install -r requirements.txt
```

3. **Chạy evaluation**:
```bash
python evaluate.py
```

## Đặc điểm nổi bật của dự án:

✅ **Tối ưu GPU**: Tự động detect và sử dụng GPU, quản lý memory thông minh  
✅ **Xử lý tiếng Việt**: Sử dụng pyvi + underthesea để tách từ chính xác  
✅ **Metrics đầy đủ**: 8 metrics khác nhau để đánh giá toàn diện  
✅ **Visualizations phong phú**: Biểu đồ, heatmap, báo cáo HTML  
✅ **Error handling**: Xử lý lỗi tốt, có thể tiếp tục khi một model lỗi  
✅ **Performance monitoring**: Theo dõi thời gian, memory usage  
✅ **Extensible**: Dễ dàng thêm models, metrics, visualization mới  

Dự án này sẽ giúp bạn đánh giá một cách khách quan và toàn diện các model embedding tiếng Việt, với giao diện người dùng thân thiện và báo cáo chi tiết!

