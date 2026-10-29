# Domain-specific residual neck với channel SE

Module mới nằm ngay sau FPN và trước FCOS detection head. Mặc định tắt để giữ baseline cũ; bật bằng `--domain-neck`.

## Kiến trúc đã triển khai

Mỗi level P3–P7 có một `DomainBottleneck` riêng, mặc định `256 → 64 → 256`. Trong một block, hai domain dùng chung `conv1`, `conv2`, `conv3` và `skip`; adapter 1×1 và SE có bộ tham số riêng cho RGB/IR.

```text
identity = skip_1x1(input)
h = ReLU(conv1_1x1(input))
u = ReLU(conv2_3x3(h) + selected_adapter_1x1(h))
z = conv3_1x1(u)
output = ReLU(selected_SE(z) + identity)
```

SE: global average pooling → Conv1×1 → ReLU → Conv1×1 → sigmoid; nhân trọng số vào từng kênh của `z`. Adapter và conv2 nhận cùng `h`, không nối tiếp nhau. Cả RGB và IR đều đi qua neck khi bật tính năng.

Theo lựa chọn của người dùng, đường skip là convolution học được và đầu ra có ReLU. Block không khởi tạo identity, không có scalar gate, correction loss hoặc attention bổ sung. Bật neck có thể thay đổi prediction ngay ở khởi tạo.

## Domain và cập nhật trọng số

| Input thực tế | Nhánh được chọn | Loss theo workflow hiện có |
|---|---|---|
| RGB, gồm cả SAGA weak/mid/high | `rgb_adapter`, `rgb_se` | GT và/hoặc KD theo route |
| IR thật | `ir_adapter`, `ir_se` | KD, không dùng GT IR train |

Tên teacher không chọn nhánh. Ví dụ route `p2_ir_flow` dùng IR teacher nhưng input là RGB qua SAGA nên cả hai forward chọn RGB. `thermal` là alias cho `ir` khi gọi module/model trực tiếp. Domain keyword mặc định là `rgb`; inference IR phải truyền `domain="ir"`.

Interface thống nhất: mọi detector adapter nhận `domain` trong `forward`; wrapper, trainer, teacher và evaluator truyền trực tiếp tham số này. Adapter/model tùy biến cũng cần nhận `domain`, không có cờ kiểm tra khả năng hỗ trợ hoặc fallback bỏ qua domain.

Student dùng cùng optimizer đang có. Nhánh không được chọn có gradient `None` và không được SGD cập nhật trong iteration đó; convolution chung và detection head vẫn nhận gradient. Attention weights là output theo input, còn tham số convolution sinh attention được optimizer học. Teacher chạy no-grad, được khởi tạo từ student và cập nhật bằng EMA hiện có trên toàn bộ model, gồm cả hai nhánh attention. Các nhánh của teacher có thể thay đổi qua EMA dù không được dùng trong forward của iteration đó.

## Chạy training

Giữ nguyên các tham số dataset, curriculum, optimizer và teacher của thí nghiệm gốc; chỉ thêm các flag neck. Ví dụ PowerShell:

```powershell
python example_flir.py --data_root 'D:\Datasets\FLIR' --output_dir './output_domain_se' --domain-neck --domain-neck-bottleneck-channels 64 --domain-neck-se-reduction 16
```

Baseline đối chứng:

```powershell
python example_flir.py --data_root 'D:\Datasets\FLIR' --output_dir './output_no_neck' --no-domain-neck
```

Đường dẫn FLIR ở trên là ví dụ, cần thay bằng dataset thực tế. Smoke không tải pretrained weights: thêm `--weights none --no-pretrained-backbone --device cpu` và giảm iteration/resize theo nhu cầu; không dùng smoke để kết luận AP.

## Evaluation và checkpoint

Validation trong training tự truyền domain theo loader. COCO evaluation tự khôi phục cấu hình neck từ checkpoint và chọn nhánh theo `--domain`:

```powershell
python evaluate_coco.py --checkpoint './output_domain_se/best_model.pth' --data-root 'D:\Datasets\FLIR' --domain ir --device cuda
```

Model gọi trực tiếp:

```python
model.eval()
with torch.no_grad():
    detections = model(images, domain="ir")
```

Resume phải dùng cùng bật/tắt neck, bottleneck channels, SE reduction và các điều kiện tương thích checkpoint hiện có. Metadata sai cấu trúc bị từ chối trước khi load state. Checkpoint baseline cũ không có cấu hình neck được hiểu là neck tắt và vẫn load được khi tắt neck. Không tự chuyển checkpoint baseline thành checkpoint neck-enabled bằng `--resume`; không dùng `strict=False` để che mismatch. Khởi tạo detector từ COCO bằng `--weights DEFAULT` vẫn được hỗ trợ, neck được khởi tạo mới.

## Phạm vi kiểm chứng

Test bao gồm phép tính của block, shape/thứ tự P3–P7, vị trí post-FPN, gradient theo domain, GT/KD routing, teacher EMA, resume, CLI và checkpoint round-trip. Chưa có kết quả huấn luyện FLIR hoặc benchmark GPU cho kiến trúc này; teacher sai vẫn có thể truyền tín hiệu KD sai. Cần đo AP từng lớp/kích thước, VRAM và latency ở cùng điều kiện baseline.

Lệnh kiểm tra toàn bộ test của dự án là `python -m pytest tests -q`. Cây `D3T/cvpods` là mã upstream tham khảo, có phụ thuộc riêng và không thuộc bộ test này. Trên Windows sandbox, các test tạo checkpoint/dataset tạm cần quyền truy cập thư mục tạm.
