import torch
print("--- KIỂM TRA GPU ORIN NANO SUPER ---")
cuda_available = torch.cuda.is_available()
print(f"CUDA khả dụng: {cuda_available}")

if cuda_available:
    print(f"Tên thiết bị: {torch.cuda.get_device_name(0)}")
    print(f"Số lượng nhân Tensor: 32")
    # Thử một phép tính ma trận trên GPU
    a = torch.ones(1000, 1000).cuda()
    b = torch.ones(1000, 1000).cuda()
    c = torch.matmul(a, b)
    print("Phép tính Tensor trên GPU: THÀNH CÔNG")
else:
    print("CẢNH BÁO: Bạn vẫn đang dùng bản Torch cho CPU (89MB là quá nhỏ).")