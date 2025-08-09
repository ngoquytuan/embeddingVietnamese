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

class VietnameseEmbeddingTester:
    def __init__(self, output_dir="embedding_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def load_document(self, file_path):
        """Load văn bản từ file .md hoặc .txt"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✅ Đã load thành công file: {file_path}")
            print(f"📄 Độ dài văn bản: {len(content)} ký tự")
            return content
        except Exception as e:
            print(f"❌ Lỗi khi load file {file_path}: {e}")
            return None
    
    def chunk_text(self, text, chunk_size=200, overlap=50):
        """Chia tài liệu thành chunks với overlap để giữ ngữ nghĩa"""
        sentences = underthesea.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_word_count = 0
        
        for sentence in sentences:
            words = ViTokenizer.tokenize(sentence).split()
            
            if current_word_count + len(words) <= chunk_size:
                current_chunk += " " + sentence
                current_word_count += len(words)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Tạo overlap bằng cách giữ lại một phần chunk cũ
                overlap_words = current_chunk.split()[-overlap:] if overlap > 0 else []
                current_chunk = " ".join(overlap_words) + " " + sentence
                current_word_count = len(overlap_words) + len(words)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_test_queries(self, chunks):
        """Tạo test queries từ chunks để đánh giá chất lượng"""
        # Lấy một số chunks ngẫu nhiên làm queries
        num_queries = min(5, len(chunks))
        query_indices = np.random.choice(len(chunks), num_queries, replace=False)
        
        queries = []
        for idx in query_indices:
            # Lấy câu đầu tiên của chunk làm query
            chunk_sentences = underthesea.sent_tokenize(chunks[idx])
            if chunk_sentences:
                queries.append({
                    'query': chunk_sentences[0],
                    'relevant_chunk_idx': idx,
                    'relevant_chunk': chunks[idx]
                })
        
        return queries
    
    def evaluate_embedding_model(self, model_name, chunks, queries):
        """Đánh giá model embedding với nhiều metrics"""
        try:
            print(f"🔄 Đang test model: {model_name}")
            start_time = time.time()
            
            # Load model
            model = SentenceTransformer(model_name)
            
            # Encode chunks
            chunk_embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
            
            results = {
                'model_name': model_name,
                'num_chunks': len(chunks),
                'encoding_time': time.time() - start_time,
                'queries_results': []
            }
            
            # Test với từng query
            for query_data in queries:
                query_embedding = model.encode([query_data['query']], convert_to_tensor=True)
                similarities = util.cos_sim(query_embedding, chunk_embeddings).cpu().numpy().flatten()
                
                # Tìm top-k similar chunks
                top_k = min(5, len(chunks))
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                query_result = {
                    'query': query_data['query'],
                    'relevant_chunk_idx': query_data['relevant_chunk_idx'],
                    'top_similarities': similarities[top_indices].tolist(),
                    'top_indices': top_indices.tolist(),
                    'relevant_chunk_rank': None,
                    'relevant_chunk_score': similarities[query_data['relevant_chunk_idx']]
                }
                
                # Tìm rank của chunk liên quan
                if query_data['relevant_chunk_idx'] in top_indices:
                    query_result['relevant_chunk_rank'] = list(top_indices).index(query_data['relevant_chunk_idx']) + 1
                
                results['queries_results'].append(query_result)
            
            # Tính metrics tổng hợp
            avg_relevant_score = np.mean([q['relevant_chunk_score'] for q in results['queries_results']])
            avg_top1_score = np.mean([q['top_similarities'][0] for q in results['queries_results']])
            
            # MRR (Mean Reciprocal Rank)
            reciprocal_ranks = []
            for q in results['queries_results']:
                if q['relevant_chunk_rank']:
                    reciprocal_ranks.append(1.0 / q['relevant_chunk_rank'])
                else:
                    reciprocal_ranks.append(0.0)
            mrr = np.mean(reciprocal_ranks)
            
            results['metrics'] = {
                'avg_relevant_score': float(avg_relevant_score),
                'avg_top1_score': float(avg_top1_score),
                'mrr': float(mrr),
                'encoding_time': results['encoding_time']
            }
            
            print(f"✅ Hoàn thành test {model_name}")
            print(f"📊 Avg relevant score: {avg_relevant_score:.4f}")
            print(f"📊 MRR: {mrr:.4f}")
            print(f"⏱️ Encoding time: {results['encoding_time']:.2f}s")
            
            return results
            
        except Exception as e:
            print(f"❌ Lỗi với model {model_name}: {e}")
            return None
    
    def run_comparison(self, document_path, models, chunk_size=200, overlap=50):
        """Chạy so sánh toàn diện các models"""
        print("🚀 Bắt đầu so sánh embedding models cho tiếng Việt")
        print("="*60)
        
        # Load document
        document = self.load_document(document_path)
        if not document:
            return None
        
        # Chunk document
        chunks = self.chunk_text(document, chunk_size=chunk_size, overlap=overlap)
        print(f"📝 Đã chia thành {len(chunks)} chunks")
        
        # Tạo test queries
        queries = self.create_test_queries(chunks)
        print(f"❓ Tạo {len(queries)} test queries")
        
        # Test từng model
        all_results = {}
        for model_name in models:
            result = self.evaluate_embedding_model(model_name, chunks, queries)
            if result:
                all_results[model_name] = result
            print("-" * 40)
        
        self.results = all_results
        return all_results
    
    def generate_report(self, save_detailed=True):
        """Tạo báo cáo so sánh chi tiết"""
        if not self.results:
            print("❌ Chưa có kết quả để tạo báo cáo")
            return
        
        print("📈 Đang tạo báo cáo...")
        
        # Tạo DataFrame cho so sánh
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name.split('/')[-1],  # Chỉ lấy tên model
                'Full_Model_Name': model_name,
                'Avg_Relevant_Score': result['metrics']['avg_relevant_score'],
                'Avg_Top1_Score': result['metrics']['avg_top1_score'],
                'MRR': result['metrics']['mrr'],
                'Encoding_Time': result['metrics']['encoding_time'],
                'Num_Chunks': result['num_chunks']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sắp xếp theo MRR (metric quan trọng nhất)
        df = df.sort_values('MRR', ascending=False)
        
        # In báo cáo console
        print("\n" + "="*80)
        print("📊 BÁO CÁO SO SÁNH EMBEDDING MODELS CHO TIẾNG VIỆT")
        print("="*80)
        
        print(f"\n🏆 RANKING (theo MRR):")
        for idx, row in df.iterrows():
            print(f"{df.index.get_loc(idx) + 1}. {row['Model']}")
            print(f"   📈 MRR: {row['MRR']:.4f}")
            print(f"   🎯 Avg Relevant Score: {row['Avg_Relevant_Score']:.4f}")
            print(f"   ⚡ Encoding Time: {row['Encoding_Time']:.2f}s")
            print()
        
        # Lưu báo cáo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV summary
        csv_path = self.output_dir / f"embedding_comparison_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 Đã lưu báo cáo CSV: {csv_path}")
        
        # JSON chi tiết
        if save_detailed:
            json_path = self.output_dir / f"detailed_results_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"💾 Đã lưu kết quả chi tiết: {json_path}")
        
        # Tạo biểu đồ so sánh
        self._create_comparison_charts(df, timestamp)
        
        return df
    
    def _create_comparison_charts(self, df, timestamp):
        """Tạo biểu đồ so sánh"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('So sánh Embedding Models cho Tiếng Việt', fontsize=16)
            
            # MRR comparison
            axes[0,0].bar(df['Model'], df['MRR'], color='skyblue')
            axes[0,0].set_title('Mean Reciprocal Rank (MRR)')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Relevant Score comparison
            axes[0,1].bar(df['Model'], df['Avg_Relevant_Score'], color='lightgreen')
            axes[0,1].set_title('Average Relevant Score')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Encoding Time comparison
            axes[1,0].bar(df['Model'], df['Encoding_Time'], color='salmon')
            axes[1,0].set_title('Encoding Time (seconds)')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Combined scatter plot
            scatter = axes[1,1].scatter(df['MRR'], df['Avg_Relevant_Score'], 
                                      s=df['Encoding_Time']*10, 
                                      c=range(len(df)), cmap='viridis', alpha=0.7)
            axes[1,1].set_xlabel('MRR')
            axes[1,1].set_ylabel('Avg Relevant Score')
            axes[1,1].set_title('Performance vs Quality (bubble size = encoding time)')
            
            # Thêm labels cho scatter plot
            for idx, row in df.iterrows():
                axes[1,1].annotate(row['Model'], 
                                 (row['MRR'], row['Avg_Relevant_Score']),
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            chart_path = self.output_dir / f"comparison_charts_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            print(f"📊 Đã lưu biểu đồ: {chart_path}")
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Không thể tạo biểu đồ: {e}")

# Sử dụng chương trình
if __name__ == "__main__":
    # Khởi tạo tester
    tester = VietnameseEmbeddingTester()
    
    # Danh sách models để test
    models_to_test = [
        "mixedbread-ai/mxbai-embed-large-v1",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "intfloat/multilingual-e5-large",
        "BAAI/bge-m3",
        # Thêm models Việt Nam nếu có
        # "vinai/phobert-base",  # Cần wrap thành sentence transformer
    ]
    
    # Chạy test (thay đổi đường dẫn file của bạn)
    document_path = "your_vietnamese_document.md"  # Thay bằng file của bạn
    
    # Nếu chưa có file, tạo file mẫu
    if not os.path.exists(document_path):
        sample_doc = """
# Văn hóa Việt Nam

Việt Nam là một quốc gia có bề dày lịch sử và văn hóa phong phú. Văn hóa Việt Nam được hình thành qua hàng nghìn năm lịch sử với những ảnh hưởng từ Trung Quốc, Ấn Độ và sau này là phương Tây.

## Kiến trúc truyền thống

Kiến trúc Việt Nam mang đậm dấu ấn của văn hóa phương Đông với những ngôi chùa, đền thờ cổ kính. Văn Miếu Quốc Tử Giám là biểu tượng của kiến trúc truyền thống Việt Nam.

## Ẩm thực

Ẩm thực Việt Nam nổi tiếng thế giới với những món ăn đặc trưng như phở, bánh mì, bún chả. Mỗi vùng miền có những đặc sản riêng biệt.

## Kinh tế hiện đại

Việt Nam đang phát triển mạnh mẽ về kinh tế với nhiều ngành công nghiệp. Công nghệ thông tin là một trong những lĩnh vực phát triển nhanh nhất.

## Du lịch

Việt Nam có nhiều điểm du lịch hấp dẫn từ Hạ Long Bay đến Hội An, thu hút hàng triệu lượt khách quốc tế mỗi năm.
        """
        
        with open(document_path, 'w', encoding='utf-8') as f:
            f.write(sample_doc)
        print(f"📝 Đã tạo file mẫu: {document_path}")
    
    # Chạy so sánh
    results = tester.run_comparison(
        document_path=document_path,
        models=models_to_test,
        chunk_size=150,  # Điều chỉnh theo nhu cầu
        overlap=30
    )
    
    if results:
        # Tạo báo cáo
        comparison_df = tester.generate_report(save_detailed=True)
        
        print("\n🎉 Hoàn thành! Kiểm tra thư mục 'embedding_test_results' để xem kết quả chi tiết.")