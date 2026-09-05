# Workflow của pipeline D3T v2 + RTC

Tài liệu này tổng hợp luồng hoạt động của code trong thư mục hiện tại, bao gồm cách khởi chạy, cách tách dataset disjoint, cách chia phase, cách chọn domain ở từng iteration và cách phối hợp teacher–student bằng EMA.

## 1. Mục đích của project

Đây là pipeline huấn luyện detector FCOS cho bài toán domain adaptation từ ảnh RGB sang ảnh thermal/IR.

Pipeline sử dụng ba nguồn dữ liệu hoặc domain:

- `rgb`: ảnh RGB có ground truth.
- `ir_gan`: ảnh RGB được chuyển thành IR giả bằng generator CycleGAN; vẫn dùng ground truth của ảnh RGB.
- `ir_real`: ảnh IR thật không nhãn; dùng pseudo-label do teacher tạo ra.

Mục tiêu là huấn luyện `student` để phát hiện ba lớp:

~~~
person, car, bicycle
~~~

## 2. Luồng khởi chạy tổng quát

~~~
bash run_train.sh
    ↓
kiểm tra Python và package
    ↓
run_train.py
    ↓
kiểm tra CUDA và resource bắt buộc
    ↓
project_config.py tạo command chuẩn
    ↓
train.py
    ↓
training_full_d3t_v2_rtc.py: main()
    ↓
load dataset, GAN, teacher, student
    ↓
training loop
    ↓
evaluation và lưu checkpoint
~~~

### Các file chính

| File | Vai trò |
|---|---|
| `run_train.sh` | Launcher dành cho server Linux; kiểm tra package rồi gọi Python launcher. |
| `run_train.py` | Preflight CUDA/resource và khởi chạy training subprocess. |
| `project_config.py` | Khai báo đường dẫn và các tham số chạy server. |
| `train.py` | Entry point ổn định, gọi `main()` của file training lớn. |
| `training_full_d3t_v2_rtc.py` | Chứa toàn bộ logic dataset, scheduler, model, loss, EMA, evaluation. |
| `third_party/pytorch-CycleGAN-and-pix2pix/models/networks.py` | Định nghĩa generator CycleGAN dùng để tạo IR giả. |
| `tests/test_server_bundle.py` | Kiểm tra đường dẫn, command, preflight và một số thành phần model. |

`run_train.sh` chạy `run_train.py`; `run_train.py` tạo command từ `project_config.py`, sau đó command gọi `train.py`. `train.py` chỉ import và gọi `main()` từ `training_full_d3t_v2_rtc.py`.

## 3. Preflight trước khi train

`run_train.py` gọi `ProjectPaths.from_root()` để xác định các đường dẫn tương đối với thư mục project.

Các resource bắt buộc:

- `data/align/align_train.txt`
- `data/align/align_validation.txt`
- `data/align/JPEGImages/`
- `data/align/Annotations/`
- `weights/latest_net_G_A.pth`
- `weights/best_epoch003_map0.2986.pth`
- `weights/best_epoch010_map0.3207.pth`
- `third_party/pytorch-CycleGAN-and-pix2pix/models/networks.py`

Nếu CUDA không khả dụng hoặc thiếu resource, launcher dừng trước khi train. Với `--check-only`, launcher chỉ thực hiện bước kiểm tra và không bắt đầu training.

CycleGAN được đóng gói trong project. Code hiện tại không tự clone CycleGAN; nếu thiếu `models/networks.py`, `require_cyclegan_repo()` sẽ báo lỗi.

## 4. Dataset và preprocessing

Dataset nằm trong `data/align/`:

~~~
data/align/
├── JPEGImages/
├── Annotations/
├── align_train.txt
└── align_validation.txt
~~~

Mỗi dòng trong file split là một stem, ví dụ:

~~~
FLIR_00258_PreviewData
~~~

Từ stem này code tìm:

~~~
FLIR_00258_RGB.jpg
FLIR_00258_PreviewData.jpeg hoặc .jpg hoặc .png
FLIR_00258_PreviewData.xml
~~~

### Dataset RGB

`FLIRRGBDataset` đọc ảnh RGB và annotation Pascal VOC XML.

- Chỉ giữ `person`, `car`, `bicycle`.
- Bỏ nhãn `FLIR`, `dog` và các nhãn không hợp lệ.
- Bỏ bounding box có diện tích nhỏ hơn `16`.
- Transform weak gồm resize về `640x640` và random horizontal flip.

Dataset trả về:

~~~
(image, target)
~~~

Trong đó `image` là tensor `[0, 1]` và `target` chứa `boxes`, `labels`.

### Dataset IR train

`FLIRIRDataset` đọc ảnh IR thật nhưng không đọc annotation. Dataset trả về tensor ảnh duy nhất:

~~~
image [3, H, W]
~~~

Ảnh IR được chuyển thành 3 channel bằng grayscale transform để phù hợp với FCOS.

### Dataset IR validation

`FLIRIRValDataset` đọc ảnh IR cùng annotation. Dataset này chỉ dùng để đánh giá, không dùng làm batch train thông thường.

### Weak và strong augmentation

- Weak transform: resize, grayscale đối với IR, horizontal flip; dùng cho teacher và supervised loss.
- Strong transform: thay đổi màu, grayscale, blur hoặc sharpness; chỉ thay đổi hình ảnh, không thay đổi tọa độ box.

FCOS tự normalize ảnh bên trong model nên code không normalize thủ công ngoài dataset.

## 5. Disjoint split là gì?

Mặc định code đọc toàn bộ `align_train.txt`, rồi tách thành hai phần không trùng stem:

~~~
align_train.txt
├── source stems → RGB source
└── target stems → real IR target
~~~

Việc này được thực hiện bởi `_split_disjoint()`.

Với `source_frac=0.5`, khoảng một nửa stem được dùng cho RGB source và phần còn lại cho IR target. Dataset hiện tại có 4.129 dòng train, nên thực tế code chia khoảng 2.064 stem cho source và 2.065 stem cho target.

Ví dụ:

~~~
FLIR_00001 → RGB source
FLIR_00002 → RGB source
FLIR_00003 → real IR target
FLIR_00004 → real IR target
~~~

Nếu `FLIR_00001` thuộc source thì pipeline có thể dùng:

~~~
RGB thật + ground truth
RGB đó chuyển qua CycleGAN thành IR giả + ground truth
~~~

nhưng không dùng ảnh IR thật tương ứng của `FLIR_00001` trong nhánh `ir_real`.

Mục đích là tránh cho source và target chứa cùng một cảnh aligned. Nếu cùng một cặp RGB–IR xuất hiện ở cả hai domain, model có thể học thuộc cảnh hoặc vị trí object, khiến kết quả domain adaptation bị lạc quan giả.

Tham số liên quan:

- `--source-frac 0.5`: tỉ lệ dữ liệu dành cho RGB source.
- `--split-seed -1`: giữ thứ tự file, lấy phần đầu làm source.
- `--split-seed >= 0`: xáo trộn xác định trước rồi mới chia.
- `--no-disjoint-split`: tắt cơ chế này; RGB và IR thật cùng dùng toàn bộ `align_train.txt`.

Validation lấy từ `align_validation.txt`, vốn đã tách khỏi training split.

## 6. Tạo ảnh IR giả bằng CycleGAN

Trong `main()`, code dựng generator:

~~~python
gan = networks.define_G(
    input_nc=3,
    output_nc=1,
    ngf=64,
    netG="resnet_9blocks",
    norm="instance",
)
~~~

Generator nạp weight từ `weights/latest_net_G_A.pth`, chuyển sang evaluation mode và đóng băng parameter.

Hàm `rgb_to_gan_ir()` thực hiện:

~~~
RGB [0, 1]
    ↓
đổi sang [-1, 1]
    ↓
CycleGAN G_A
    ↓
IR giả 1 channel [-1, 1]
    ↓
đổi lại [0, 1]
    ↓
repeat thành 3 channel
~~~

Bounding box của ảnh RGB được giữ nguyên khi chuyển sang `ir_gan`.

## 7. Phase và domain schedule

Lịch mặc định được khai báo ở `DEFAULT_RATIO_SCHEDULE`:

~~~
7:3:0,5:4:1,4:3:3,3:2:5,2:1:7,2:0:8
~~~

Mỗi tuple có dạng:

~~~
RGB : IR-GAN : IR-real
~~~

Với cấu hình server `18.000 iterations` và 6 phase, mỗi phase dài khoảng 3.000 iterations:

| Phase | Iteration gần đúng | RGB | IR-GAN | IR-real |
|---:|---:|---:|---:|---:|
| 0 | 1–3.000 | 7 | 3 | 0 |
| 1 | 3.001–6.000 | 5 | 4 | 1 |
| 2 | 6.001–9.000 | 4 | 3 | 3 |
| 3 | 9.001–12.000 | 3 | 2 | 5 |
| 4 | 12.001–15.000 | 2 | 1 | 7 |
| 5 | 15.001–18.000 | 2 | 0 | 8 |

Tỉ lệ này tính trên số iteration trong phase, không phải ba batch trong cùng một iteration. Mỗi iteration chỉ lấy một batch từ một domain.

### `phase_for_iter()`

Hàm này xác định phase dựa trên iteration zero-based:

~~~python
phase = min(
    n_phases - 1,
    it * n_phases // total_iters,
)
~~~

### `get_domain()`

`BridgeDomainScheduler.get_domain()` trả về đúng một trong ba chuỗi:

~~~
"rgb"
"ir_gan"
"ir_real"
~~~

Nó lưu ba trạng thái chính:

- `_weights`: tỉ lệ mong muốn, ví dụ `rgb=0.7`, `ir_gan=0.3`.
- `_alloc`: số lần mỗi domain đã được chọn trong phase hiện tại.
- `_k`: số lần `get_domain()` đã được gọi trong phase.

Mỗi lần gọi, code tính độ thiếu hụt:

~~~python
desired_count = _weights[domain] * _k
deficit = desired_count - _alloc[domain]
~~~

Domain có `deficit` lớn nhất được chọn. Đây là cách error-diffusion để xen kẽ domain và giữ số lần chọn gần với ratio mong muốn.

Khi sang phase mới, `_enter_phase()` reset `_alloc` và bắt đầu phân phối lại theo ratio mới. Domain có ratio bằng 0 bị loại khỏi danh sách active nên không bao giờ được chọn trong phase đó.

## 8. Một iteration lấy bao nhiêu batch?

Mỗi training iteration lấy đúng một batch:

~~~python
domain = domain_scheduler.get_domain(it - 1)
~~~

Sau đó:

~~~python
if domain in ("rgb", "ir_gan"):
    images, targets = next(rgb_gen)
else:
    images = next(ir_gen)
~~~

Với `batch_size=8`, một iteration thường xử lý 8 ảnh của một domain.

- `rgb`: một batch RGB có GT.
- `ir_gan`: một batch RGB, sau đó cả batch được chuyển qua GAN.
- `ir_real`: một batch IR thật không nhãn.

`infinite(loader)` làm cho DataLoader quay lại từ đầu khi hết dữ liệu, vì training được điều khiển bằng số iteration chứ không phải số epoch.

## 9. Khởi tạo teacher và student

Code tạo ba detector FCOS cùng kiến trúc:

~~~
t_model_rgb
t_model_ir
s_model
~~~

- `t_model_rgb` nạp checkpoint RGB.
- `t_model_ir` nạp checkpoint IR-GAN.
- `s_model` khởi tạo bằng checkpoint RGB.

Teacher có `requires_grad=False`, nghĩa là không được cập nhật bằng optimizer. Student là model duy nhất nhận gradient từ detection loss, consistency loss và có thể thêm RTC loss.

Teacher vẫn thay đổi trong training vì được cập nhật thủ công bằng EMA.

## 10. Điều phối teacher trong từng domain

| Domain hiện tại | Teacher cùng domain (`t_match`) | Teacher cross-domain (`t_cross`) | Teacher EMA |
|---|---|---|---|
| `rgb` | `t_model_rgb` | `t_model_ir` | `t_model_rgb` |
| `ir_gan` | `t_model_ir` | `t_model_rgb` | `t_model_ir` |
| `ir_real` | `t_model_ir` | `t_model_rgb` | `t_model_ir` |

Teacher chạy trên ảnh weak trong `torch.no_grad()` để tạo hai bộ pseudo-label:

~~~
pseudo_match = dự đoán từ teacher cùng domain
pseudo_cross = dự đoán từ teacher cross-domain
~~~

Sau đó `filter_pseudo_labels()` chỉ giữ các prediction có confidence từ `pseudo_conf` trở lên, mặc định là `0.5`.

Student nhận cùng ảnh sau strong augmentation và tính hai consistency loss:

~~~python
loss_uns_match = sum(s_model(s_images, pseudo_match).values())
loss_uns_cross = sum(s_model(s_images, pseudo_cross).values())
~~~

Trọng số mặc định:

~~~
w_match = 1.0
w_cross = 0.3
~~~

Teacher cùng domain được ưu tiên cao hơn teacher cross-domain.

## 11. Cập nhật EMA

Hàm EMA nằm ở `update_ema()`:

~~~python
teacher = alpha * teacher + (1 - alpha) * student
~~~

Code thực hiện công thức trên từng parameter tương ứng của teacher và student:

~~~python
for t_p, s_p in zip(teacher.parameters(), student.parameters()):
    t_p.data.mul_(alpha).add_(
        s_p.data,
        alpha=1.0 - alpha,
    )
~~~

Sau mỗi optimizer step, training loop gọi:

~~~python
update_ema(ema_teacher, s_model, alpha=ema_alpha)
~~~

Chu trình là:

~~~
teacher dự đoán pseudo-label
        ↓
student tính loss
        ↓
backward + optimizer.step()
        ↓
student cập nhật teacher tương ứng bằng EMA
        ↓
iteration tiếp theo dùng teacher mới
~~~

Chỉ teacher tương ứng với domain hiện tại được cập nhật:

- `rgb` cập nhật `t_model_rgb`.
- `ir_gan` và `ir_real` cập nhật `t_model_ir`.

`thermal_head` là module phụ riêng nên không tham gia EMA.

## 12. Loss trong từng domain

### RGB và IR-GAN

Hai domain này có ground truth:

~~~
total_loss
    = supervised_loss
    + lambda × consistency_loss
    + rtc_lambda × rtc_loss (chỉ IR-GAN)
~~~

`supervised_loss` là tổng các loss do FCOS trả về khi student chạy trên ảnh weak và GT thật.

### IR thật

IR thật không có GT:

~~~
total_loss
    = lambda × consistency_loss
    + rtc_lambda × rtc_loss
~~~

Consistency loss là:

~~~python
loss_uns = (
    w_match * loss_uns_match
    + w_cross * loss_uns_cross
)
~~~

`lambda` được ramp tuyến tính từ 0 đến `lambda_max` bằng `lambda_for_iter()`. Server dùng `lambda_max=1.0` và `lambda_ramp_iters=5400`.

## 13. RTC loss

RTC là loss phụ dành cho domain thermal, được triển khai trong `rtc_loss()`.

Nó so sánh relative contrast giữa object và background trong ảnh thermal với response map do `ThermalResponseHead` sinh ra từ feature FPN của student.

~~~
ảnh thermal weak + box
        ↓
đo contrast object/background trong ảnh
        ↓
student backbone + FPN
        ↓
ThermalResponseHead
        ↓
response map
        ↓
RTC ranking loss
~~~

RTC:

- Không áp dụng cho `rgb`.
- Với `ir_gan`, dùng ground truth box.
- Với `ir_real`, dùng pseudo-label của teacher IR.
- Chỉ bật từ `rtc_start_phase`, server mặc định là phase 2 (`4:3:3`).
- Có thể tắt bằng `--disable-rtc` hoặc `--rtc-weight 0`.

Thermal head gồm:

~~~
Conv 3x3 → ReLU → Conv 1x1 → response map 1 channel
~~~

Thermal head được cập nhật bằng optimizer cùng student, nhưng checkpoint của nó chỉ cần cho việc tiếp tục phân tích/training RTC; inference detector chỉ cần student checkpoint.

## 14. Một iteration đầy đủ

~~~
1. Scheduler xác định phase và chọn domain.

2. Lấy đúng một batch từ loader tương ứng.

3. Nếu là ir_gan, chuyển batch RGB qua CycleGAN.

4. Chọn teacher cùng domain và teacher cross-domain.

5. Teacher dự đoán trên ảnh weak trong no_grad().

6. Lọc pseudo-label theo confidence threshold.

7. Tạo ảnh strong augmentation cho student.

8. Tính supervised loss nếu có GT.

9. Tính consistency loss từ hai teacher.

10. Tính RTC loss nếu domain và phase cho phép.

11. Cộng các loss thành total_loss.

12. optimizer.zero_grad().

13. total_loss.backward().

14. Gradient clipping.

15. optimizer.step().

16. Cosine learning-rate scheduler step.

17. EMA cập nhật teacher tương ứng.

18. Ghi log; nếu đến kỳ thì evaluation và lưu checkpoint.
~~~

## 15. Evaluation và output

Mỗi `eval_every=1000` iterations, code đánh giá student trên `FLIRIRValDataset` bằng `MeanAveragePrecision`.

Các metric:

- mAP từ IoU `0.50:0.95`.
- mAP tại IoU `0.50`.
- mAP tại IoU `0.75`.
- mAP theo từng class.

Nếu mAP tốt hơn trước, lưu:

~~~
outputs/best_iterXXXXXX_mapX.XXXX.pth
~~~

Cuối training lưu:

~~~
outputs/last_iterXXXXXX.pth
outputs/thermal_head_last.pth   # nếu RTC bật
~~~

Nếu visualization được bật, code lưu ảnh validation có:

- Ground truth màu xanh.
- Prediction màu đỏ.

## 16. Cấu hình server thực tế

Khi chạy qua `run_train.sh`, `project_config.py` truyền:

~~~text
total-iters       = 18000
eval-every        = 1000
img-size          = 640
batch-size        = 8
lambda-ramp-iters = 5400
ema-alpha         = 0.999
rtc-weight        = 0.1
rtc-start-phase   = 2
rtc-fpn-level     = 0
~~~

Nếu chạy trực tiếp `training_full_d3t_v2_rtc.py` mà không truyền argument, một số default khác được dùng, ví dụ `total-iters=20000`, `ema-alpha=0.9996` và `rtc-fpn-level=1`.

## 17. Tóm tắt ngắn gọn

~~~
RGB có nhãn
   ├── train trực tiếp
   └── qua CycleGAN thành IR giả

IR thật không nhãn
   └── nhận pseudo-label từ teacher

Scheduler
   └── mỗi iteration chọn đúng một domain

Teacher
   └── tạo pseudo-label, không backprop

Student
   └── học bằng supervised + consistency + RTC

EMA
   └── làm mượt student để cập nhật teacher

Evaluation
   └── đo trên IR validation và lưu student checkpoint
~~~
