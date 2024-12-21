import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

st.title("Trực Quan Hóa Ranh Giới Phân Loại của KNN")

st.markdown("""
Ứng dụng này cho phép bạn điều chỉnh giá trị **K** trong **K-Nearest Neighbors (KNN)** và trực quan hóa ranh giới phân loại tương ứng.
""")

@st.cache(allow_output_mutation=True)
def generate_data(n_samples_per_class=100):
    np.random.seed(0)
    X_A = np.random.randn(n_samples_per_class, 2) + np.array([2, 2])
    X_B = np.random.randn(n_samples_per_class, 2) + np.array([5, 5])
    X = np.vstack((X_A, X_B))
    y = np.array(['A'] * n_samples_per_class + ['B'] * n_samples_per_class)
    return X, y

X, y = generate_data()

if st.checkbox("Hiển thị dữ liệu"):
    st.subheader("Dữ Liệu")
    st.write("Tổng số điểm dữ liệu:", X.shape[0])
    st.write("Một vài điểm dữ liệu mẫu:")
    st.write({
        'X1': X[:, 0],
        'X2': X[:, 1],
        'Y': y
    })

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 'A' -> 0, 'B' -> 1

st.sidebar.header("Thông Số KNN")
K = st.sidebar.slider("Chọn giá trị K", min_value=1, max_value=100, value=5, step=1)

if K > X.shape[0]:
    st.sidebar.error(f"Giá trị K không được lớn hơn số lượng điểm dữ liệu ({X.shape[0]}).")
    st.stop()

# Huấn luyện mô hình KNN
distance_metric = st.sidebar.selectbox("Chọn loại khoảng cách", ["euclidean", "manhattan", "minkowski"])
knn = KNeighborsClassifier(n_neighbors=K, metric=distance_metric)
knn.fit(X, y_encoded)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
resolution = 300
xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                     np.linspace(y_min, y_max, resolution))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = knn.predict(grid)
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_encoded, cmap=plt.cm.Paired, edgecolor='k', s=30)
ax.set_title(f'KNN Decision Boundary (K={K})')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

handles = []
for i, cls in enumerate(le.classes_):
    handles.append(plt.Line2D([0], [0], marker='o', color='w', label=cls,
                              markerfacecolor=plt.cm.Paired(i), markersize=10))
ax.legend(handles=handles, title='Classes')

st.pyplot(fig)

st.subheader("Thông Tin Về Mô Hình KNN")
st.write(f"**Giá trị K:** {K}")
st.write(f"**Số lượng điểm dữ liệu:** {X.shape[0]}")
st.write(f"**Số lớp:** {len(le.classes_)}")
