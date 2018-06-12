# Nén đơn câu sử dụng kiến trúc Attention kết hợp Bidirectional Long Short-Term Memory
```
@inproceedings{tran2016effective,
  title={Effective attention-based neural architectures for sentence compression with bidirectional long short-term memory},
  author={Tran, Nhi-Thao and Luong, Viet-Thang and Nguyen, Ngan Luu-Thuy and Nghiem, Minh-Quoc},
  booktitle={Proceedings of the Seventh Symposium on Information and Communication Technology},
  pages={123--130},
  year={2016},
  organization={ACM}
}
```
Tác giả: Trần Thị Thảo Nhi

Đơn vị: Bộ môn Công nghệ Tri Thức, Đại học Khoa học Tự Nhiên Tp. HCM

Email: tttnhi@mso.hcmus.edu.vn

Đề tài SKHCN: Xây dựng công cụ tổng hợp tin tức tiếng Việt và ứng dụng

#### Mô tả
Chương trình nén câu (nén đơn câu) rút gọn câu bằng cách bỏ đi các thành phần thừa trong câu đầu vào.
Chương trình nén câu của nhóm áp dụng kiến trúc học sâu với mạng Attention, trong đó mỗi một từ của câu đầu vào được học và được gán nhãn ``0`` (từ thừa cần loại bỏ) hoặc ``1`` (từ quan trọng cần được giữ lại).
Mạng Attention gán nhãn dựa trên việc học ngữ cảnh của câu, tập trung vào ngữ cảnh trực tiếp của từ đang được dự đoán.
Điểm mạnh của mạng Attention là tại mỗi thời điểm việc học chỉ tập trung vào những ngữ cảnh có liên quan đến từ đang xét, việc này sẽ giảm nhiễu và tăng độ chính xác cho quá trình gán nhãn.
Kết quả trả về của chương trình nén câu là câu nén ngắn gọn chứa các nội dung quan trọng của câu đầu vào.

#### Cách dùng
###### Nén câu
```python
   python compress-1.py <input> <output> <model> <dataset>
```
 trong đó:
  + input chứa câu cần nén
  + output chứa câu nén
  + model chứa tham số mô hình đã được train
  + dataset nén câu ([*link*](https://github.com/nhittt/VCC))
 
###### Train mô hình
 ```python
    python sentencecompessionVietnamese.py <dataset>
 ```
