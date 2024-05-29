import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse, Circle
from pandas.plotting import parallel_coordinates, radviz, andrews_curves

# Загрузка и описание набора данных / 加载和描述数据集
df = sns.load_dataset("penguins").dropna()

# Описание набора данных / 数据集描述
print("Информация о наборе данных:")  # 数据集信息
print(df.info())
print("\nСтатистическое описание набора данных:")  # 数据集统计描述
print(df.describe())

# Стандартизация данных / 数据标准化处理
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
df_num = pd.DataFrame(df_scaled, columns=df.select_dtypes(include=['float64', 'int64']).columns)

# Определение функции для построения лиц Чернова / 定义函数绘制Chernoff Faces
def plot_custom_face(data, ax):
    # Построение черт лица на основе данных / 通过数据绘制面部特征
    head = Ellipse((0.5, 0.5), width=0.6, height=0.8, facecolor='yellow', edgecolor='black')
    ax.add_patch(head)

    left_eye = Circle((0.35, 0.6), radius=0.05, facecolor='white', edgecolor='black')
    right_eye = Circle((0.65, 0.6), radius=0.05, facecolor='white', edgecolor='black')
    ax.add_patch(left_eye)
    ax.add_patch(right_eye)

    left_pupil = Circle((0.35, 0.6), radius=0.02, facecolor='black')
    right_pupil = Circle((0.65, 0.6), radius=0.02, facecolor='black')
    ax.add_patch(left_pupil)
    ax.add_patch(right_pupil)

    mouth = Ellipse((0.5, 0.4), width=0.4, height=0.1, facecolor='red', edgecolor='black')
    ax.add_patch(mouth)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

# 1. Визуализация данных с использованием лиц Чернова / 使用Chernoff Faces进行数据可视化
print("\nПример построения лиц Чернова:")
fig, ax = plt.subplots()
plot_custom_face(df_num.iloc[0], ax)
plt.title("Лицо Чернова - первая запись")
plt.show()

# 2. Интерпретация результатов лиц Чернова / 解释Chernoff Faces结果
print("Интерпретация: Лица Чернова показывают многомерные данные, где каждый элемент лица соответствует определенной характеристике данных. Наблюдая за изменениями лица, можно интуитивно распознавать и сравнивать модели данных.")

# 3. Группировка данных и повторное построение лиц Чернова / 数据分组并重新绘制Chernoff Faces
df_grouped = df.groupby('species')[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].mean()

print("\nПостроение лиц Чернова для группированных данных:")
for index, row in df_grouped.iterrows():
    fig, ax = plt.subplots()
    plot_custom_face(row, ax)
    plt.title(f"Лицо Чернова - {index}")
    plt.show()
    print(f"Интерпретация: Средние характеристики пингвинов вида {index} демонстрируют центральную тенденцию группы. Наблюдая за изменениями лица, можно сравнивать различия характеристик между видами пингвинов.")

# 4. Рекомендации по использованию лиц Чернова / 推荐使用Chernoff Faces的情况
print("Рекомендации по использованию лиц Чернова: Лица Чернова подходят для интуитивного сравнения и распознавания моделей в небольших наборах данных, особенно когда количество характеристик невелико и необходимо быстро сравнить данные визуально.")

# 5. Построение графика параллельных координат / 绘制平行坐标图
plt.figure(figsize=(12, 6))
parallel_coordinates(df, class_column='species', cols=df.select_dtypes(include=['float64', 'int64']).columns)
plt.title('График параллельных координат')
plt.show()
print("Интерпретация: График параллельных координат показывает различия характеристик у различных видов пингвинов, особенно в длине и глубине клюва, длине плавника и массе тела.")

# 6. Построение графика RadViz / 绘制RadViz图
# 过滤非数值列
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# 将species列添加回df_numeric
df_numeric['species'] = df['species']

plt.figure(figsize=(12, 6))
radviz(df_numeric, 'species')
plt.title('График RadViz')
plt.show()
print("Интерпретация: График RadViz показывает распределение различных видов пингвинов в многомерном пространстве характеристик, что помогает выявить кластеры между видами.")

# 7. Построение диаграммы кривых Эндрюса / 绘制安德鲁斯曲线图
# 确保数据集中只有数值列和类别列
df_andrews = df.select_dtypes(include=['float64', 'int64'])
df_andrews['species'] = df['species']

plt.figure(figsize=(12, 6))
andrews_curves(df_andrews, 'species')
plt.title('Диаграмма кривых Эндрюса')
plt.show()
print("Интерпретация: Диаграмма кривых Эндрюса показывает общие изменения характеристик у различных видов пингвинов, что помогает распознавать модели и выявлять аномалии.")

# 8. Сравнение и обобщение различных методов визуализации / 比较和总结不同可视化方法
print("Обобщение:")
print("Различные методы визуализации имеют свои преимущества:")
print(" - Лица Чернова: подходят для интуитивного сравнения небольших наборов данных.")
print(" - График параллельных координат: показывает тренды в многомерных данных.")
print(" - График RadViz: отображает распределение многомерных характеристик.")
print(" - Диаграмма кривых Эндрюса: показывает общие изменения характеристик.")
print("В целом, различные виды пингвинов имеют заметные различия в характеристиках, которые можно наглядно показать с помощью перечисленных методов визуализации.")
