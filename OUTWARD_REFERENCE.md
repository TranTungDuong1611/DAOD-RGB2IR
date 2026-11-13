# Outward reference cho domain SE neck

OutRef được bật riêng bằng `--domain-neck --out-ref`. Nó nằm trong hai block
post-FPN P3/P4; P5–P7 tiếp tục dùng adapter + channel SE hiện có. Mỗi level có
một OutRef riêng, dùng chung trọng số cho RGB/IR để bắt đầu học từ phase 1.
Adapter và SE vẫn có trọng số riêng theo domain.

## Đối chiếu với code BiRefNet

Nguồn tham chiếu chính thức:
[models/birefnet.py](https://github.com/ZhengPeng7/BiRefNet/blob/main/models/birefnet.py)
(`gdt_convs`, `gdt_convs_pred`, `gdt_convs_attn`, `gradient_func`) và
[train.py](https://github.com/ZhengPeng7/BiRefNet/blob/main/train.py)
(gradient loss).

Giữ cấu trúc Conv3×3 → BatchNorm → ReLU tạo gradient feature, sau đó tách hai
Conv1×1: một dự đoán biên, một sinh spatial attention. Chúng nhận cùng feature,
không lấy edge prediction nhân trực tiếp vào feature. Nhánh dự đoán biên chỉ
chạy khi train; nhánh attention tiếp tục chạy khi evaluate/inference.

```text
F = feature FPN của một level
h = ReLU(conv1(F))
u = ReLU(conv2(h) + adapter_domain(h))
z = conv3(u)

gradient_feature = ReLU(BN(Conv3x3(z)))
edge_logits = Conv1x1_pred(gradient_feature)       # chỉ train
spatial_gain = 2 * sigmoid(Conv1x1_attn(gradient_feature))

output = ReLU(SE_domain(z) * spatial_gain + skip_1x1(F))
```

Những điều chỉnh có chủ đích cho thí nghiệm detection:

- BiRefNet dùng `sigmoid` cho attention. Bản này dùng `2 * sigmoid`, khởi tạo
  weight/bias đầu attention bằng 0, nên gain ban đầu bằng 1. OutRef không làm
  thay đổi output của **cùng block SE có cùng trọng số** tại khởi tạo. Toàn bộ
  neck vẫn có skip Conv1×1 học được và ReLU, không khởi tạo thành identity.
- BiRefNet lấy Laplacian kernel 5 trên ảnh trung bình kênh và dùng mask liên quan
  đến segmentation prediction. Bản detection dùng trị tuyệt đối Laplacian 5×5,
  chuẩn hóa theo giá trị lớn nhất mỗi ảnh hợp lệ về [0, 1], rồi dùng ROI box.
  Kernel chuẩn hóa L1 được viết bằng PyTorch, không thêm dependency Kornia.
- Code loss BiRefNet nội suy gradient prediction về kích thước target rồi dùng
  BCE trên các giá trị qua sigmoid. Bản này giữ nội suy bilinear
  `align_corners=True`, dùng `BCEWithLogits` với target biên [0, 1] đã chuẩn hóa.
  Đây là adaptation cho detection, không phải tái hiện nguyên xi loss segmentation.

## Target, vùng giám sát và gradient

Target lấy từ đúng ảnh **student** sau SAGA/photometric augmentation và resize.
Adapter cung cấp mean/std thực tế để hoàn nguyên normalize trước khi lấy trung
bình kênh. Laplacian tính trên vùng ảnh thật với reflect padding; ảnh quá nhỏ
dùng replicate. Batch padding không tham gia target hoặc loss.

| Route | Vùng tính auxiliary loss |
|---|---|
| RGB có GT, gồm phase 1 | Box GT sau cùng phép resize của student |
| Phase 2 `p2_ir_flow` | Box GT RGB trên ảnh SAGA-high, qua adapter/SE IR |
| IR thật phase 3/4 | Box detection của teacher đang được dùng cho KD, sau NMS và lọc score |

Box chỉ là mask chọn pixel; không vẽ cạnh box làm nhãn biên. Box teacher từ
postprocess ở tọa độ ảnh gốc được scale sang tọa độ student. Score là score
detection của FCOS (`sqrt(sigmoid(cls) * sigmoid(quality))`), khác ngưỡng HM
chọn điểm cho KD. Target và box đều detach; không dùng GT IR train.

Loss tính trung bình trên pixel ROI mỗi ảnh, rồi trên các ảnh có ROI hợp lệ và
hai level. Ngoài ROI/padding không có loss trực tiếp. Ảnh không có box đủ
confidence bị bỏ qua; batch không có ROI trả zero có nối graph.
`loss_out_ref` trong log là giá trị **đã nhân weight**.

```text
L_total = L_supervised_đã_weight + L_KD_đã_weight + lambda_out_ref * L_gradient
```

Auxiliary loss cập nhật gradient feature, đầu dự đoán biên và các lớp phía
trước `z`. Đầu attention học qua detection/KD loss; hai đầu cùng chia sẻ
gradient feature. Đầu dự đoán biên không cập nhật SE hoặc skip phía sau `z`.
Teacher chạy eval/no-grad, tái sử dụng output KD hiện có để chọn box; không
forward teacher/student thêm lần nữa. EMA hiện tại cập nhật cả OutRef và
BatchNorm buffers. Copy RGB → IR chỉ copy adapter/SE như trước; OutRef dùng
chung hai domain nên không cần copy giữa domain.

## Cấu hình và lệnh train

| Flag | Mặc định | Tác dụng |
|---|---|---|
| `--out-ref` | tắt | Bật OutRef, yêu cầu `--domain-neck` |
| `--out-ref-channels` | 16 | Số kênh gradient feature ở mỗi level |
| `--no-out-ref-batch-norm` | không đặt | Bỏ BatchNorm cho ablation |
| `--out-ref-loss-weight` | 0.1 | Weight auxiliary loss; 0 giữ attention nhưng tắt loss biên |
| `--out-ref-teacher-score-thresh` | 0.7 | Score tối thiểu của box teacher trên ảnh không GT |

Các giá trị 0.1 và 0.7 là điểm bắt đầu thí nghiệm, chưa được xác nhận tốt nhất
trên FLIR. ROI có cả nền/texture trong box; pseudo-box sai vẫn có thể chọn vùng
sai. BatchNorm dùng chung có thể chịu domain shift; có flag bỏ BN để đối chứng.

Lệnh Linux batch size 8, giữ 20% RGB phase 4:

```bash
mkdir -p ./outputs/domain_se_outref_b8_run01
nohup python -u -B example_flir.py \
  --data_root /home/intern/D3T/data/align \
  --output_dir ./outputs/domain_se_outref_b8_run01 \
  --workflow curriculum --total_iters 10000 \
  --batch_size 8 --eval_batch_size 8 --workers 4 --device cuda \
  --domain-neck --out-ref --out-ref-channels 16 \
  --out-ref-loss-weight 0.1 --out-ref-teacher-score-thresh 0.7 \
  --phase4-rgb-ratio 0.2 --eval_every 500 --log_interval 1 \
  > ./outputs/domain_se_outref_b8_run01/nohup.log 2>&1 &
```

Đối chứng trên cùng branch: bỏ `--out-ref` để chạy SE neck hiện có; giữ
`--out-ref` nhưng đặt `--out-ref-loss-weight 0` để đo riêng ảnh hưởng attention.
Dùng cùng data split, seed và lịch train khi so sánh mAP theo phase. Lệnh trên
không bật pseudo-label audit; việc chọn box OutRef không cần bật audit.

## Pretrained, resume và evaluation

`--weights DEFAULT` vẫn load pretrained detector trước khi gắn neck/OutRef.
OutRef khởi tạo mới, không thay hoặc reset backbone/FPN pretrained. Trọng số
detector vẫn chịu fine-tuning qua các loss như trước.

Checkpoint lưu cấu hình OutRef, model, teacher, optimizer và BN state. Resume
cần cùng `--domain-neck`, bật/tắt OutRef, số kênh và cấu hình BN; metadata sai
kiến trúc bị từ chối trước khi load tensor. Checkpoint SE-only cũ vẫn dùng được
khi OutRef tắt. Không dùng `--resume` checkpoint SE-only cho run bật OutRef;
đó là thay kiến trúc, không phải resume. Chưa thêm cơ chế migrate checkpoint.

`evaluate_coco.py` tự dựng OutRef từ checkpoint, không cần flag OutRef:

```bash
python evaluate_coco.py \
  --checkpoint ./outputs/domain_se_outref_b8_run01/best_model.pth \
  --data-root /home/intern/D3T/data/align --domain ir --device cuda
```

## Kiểm chứng

```bash
python -m pytest tests -q -p no:cacheprovider --tb=short
```

Test gồm công thức block, neutral gain, gradient tách hai đầu, target/mask,
tọa độ box, cập nhật theo domain, một forward mỗi model, EMA, checkpoint
round-trip, resume và CLI smoke qua bốn phase với FCOS backbone nhỏ trên CPU.
Smoke không phải kết quả train FLIR thực; chưa đo mAP, VRAM hay tốc độ GPU.
