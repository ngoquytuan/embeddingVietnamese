import os
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer
import huggingface_hub

# Hàm lấy đường dẫn cache của Hugging Face
def get_cache_dir():
    cache_dir = os.getenv('HF_HOME', os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    return Path(cache_dir)

# Hàm liệt kê tất cả các mô hình đã tải về
def list_cached_models():
    cache_dir = get_cache_dir()
    print("Danh sách các mô hình đã tải về:")
    model_count = 0
    for model_dir in cache_dir.glob("models--*"):
        model_name = model_dir.name.replace("models--", "").replace("--", "/")
        size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / (1024 ** 3)  # Kích thước (GB)
        print(f"- Mô hình: {model_name}, Đường dẫn: {model_dir}, Kích thước: {size:.2f} GB")
        model_count += 1
    if model_count == 0:
        print("Không tìm thấy mô hình nào trong cache.")

# Hàm kiểm tra chi tiết một mô hình cụ thể
def get_model_details(model_name):
    cache_dir = get_cache_dir()
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    model_path = cache_dir / model_dir_name
    if model_path.exists():
        size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / (1024 ** 3)  # Kích thước (GB)
        print(f"Chi tiết mô hình: {model_name}")
        print(f"- Đường dẫn: {model_path}")
        print(f"- Kích thước: {size:.2f} GB")
        return model_path
    else:
        print(f"Mô hình {model_name} không tồn tại trong cache.")
        return None

# Hàm tải một mô hình mới
def download_model(model_name):
    try:
        print(f"Đang tải mô hình {model_name}...")
        model = SentenceTransformer(model_name)
        print(f"Mô hình {model_name} đã được tải về thành công!")
        get_model_details(model_name)
    except Exception as e:
        print(f"Lỗi khi tải mô hình {model_name}: {str(e)}")

# Hàm xóa một mô hình khỏi cache
def delete_model(model_name):
    cache_dir = get_cache_dir()
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    model_path = cache_dir / model_dir_name
    if model_path.exists():
        shutil.rmtree(model_path)
        print(f"Mô hình {model_name} đã được xóa khỏi cache.")
    else:
        print(f"Mô hình {model_name} không tồn tại trong cache.")

# Hàm chính để tương tác với người dùng
def main():
    while True:
        print("\n=== Quản lý mô hình SentenceTransformers ===")
        print("1. Liệt kê các mô hình đã tải")
        print("2. Kiểm tra chi tiết một mô hình")
        print("3. Tải một mô hình mới")
        print("4. Xóa một mô hình")
        print("5. Thoát")
        choice = input("Nhập lựa chọn (1-5): ")

        if choice == "1":
            list_cached_models()
        elif choice == "2":
            model_name = input("Nhập tên mô hình (ví dụ: dangvantuan/vietnamese-embedding): ")
            get_model_details(model_name)
        elif choice == "3":
            model_name = input("Nhập tên mô hình để tải (ví dụ: dangvantuan/vietnamese-embedding): ")
            download_model(model_name)
        elif choice == "4":
            model_name = input("Nhập tên mô hình để xóa (ví dụ: dangvantuan/vietnamese-embedding): ")
            delete_model(model_name)
        elif choice == "5":
            print("Thoát chương trình.")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")

if __name__ == "__main__":
    main()