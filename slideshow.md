---
marp: true
theme: uncover
paginate: true
class:
    - lead
    - invert
size: 4:3
header: "VNU - Hanoi University of Science"
footer: "Nguyen Manh Linh - Dao Thi Thu Hong"
output:
    pdf_document:
        css: style.css
---

<style>
    section {
        font-size: 25px;
    }
</style>

## Dự đoán giá nhà
### Mô hình Linear Regression

---

## Dữ liệu đầu vào

- Kc_house_data.csv - https://www.kaggle.com/datasets/swathiachath/kc-housesales-data

- 21 variables and 21613 observations

- Xử lý dữ liệu:
    - Bỏ các trường: id, datetime, vị trí (chưa phân loại)
    - Thêm các trường: years_built, years_renovated
    - Chuẩn hóa $X = (X - X_{min}) / (X_{max} - X_{min})$
    - Scale: $y = y/1e6$

---
## Chuẩn bị dữ liệu

- $X$: ma trận $N \times 16$ (thêm 1 cột cho hệ số tự do)

- $y$: vector $N \times 1$

- Chia tập train - test theo tỉ lệ $80/20$

---
## Loss function, Gradient, Hessian

- Loss function: $f(w) = 1/2 \lVert y - Xw \rVert ^2$

- Gradient: $\nabla f(w) = X^T X w - X^T y$

- Hessian: $H = X^T X$

---
## Thuật toán GD

- Công thức cập nhật: $w^{(+)} = w - t \nabla f(w)$

- Backtracking: 
    - Với $f(w^{(+)}) > f(w) - t \nabla f(w)$
    - Lấy $t = \beta t$ với $\beta < 1$ và lặp lại

---
## Thuật toán Accelerated GD

- Chọn điểm bắt đầu $w^{(0)} = w^{(-1)}$, lặp lại:

    - $v = w^{(k - 1)} + \frac{k-2}{k+1} (w^{(k - 1)} - w^{(k - 2)})$

        $w^{(k)} = v - t_{k}\nabla f(v)$

        với $k = 1,2,3,...$

    - Backtracking:
        - Với $f(w^{(+)}) > f(v) + \nabla f(v)^T (w^{(+)} - v) + \frac{1}{2t} \lVert w^{(+)} - v \rVert ^2$

        - Lấy $t = \beta t$ ($\beta < 1$) và  $w^{(+)} = v - t \nabla f(v)$, lặp lại

---
## Thuật toán Newton
