import torch
print(torch.cuda.is_available())  # Nếu True nghĩa là có GPU
print(torch.cuda.device_count())  # Số GPU có thể dùng
print(torch.cuda.get_device_name(0))  # Tên GPU
print(torch.version.cuda)


# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng: {device}")

# Tạo tensor và đưa lên GPU
x = torch.randn(3, 3).to(device)
print(x)

# Khi train model, hãy đảm bảo rằng:
# Chuyển dữ liệu và mô hình lên GPU:
# model.to(device)
# inputs, labels = inputs.to(device), labels.to(device)

# Nếu muốn kiểm tra tài nguyên GPU khi train, bạn có thể mở terminal và chạy
# watch -n 1 nvidia-smi


