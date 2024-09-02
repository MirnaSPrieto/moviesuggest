# Importar librerías necesarias
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import LabelEncoder

# Configuraciones iniciales
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)
sns.set(style="whitegrid")

# Descargar y extraer el dataset (suponiendo que ya tienes el archivo)
with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

# Cargar datos
df_movies = pd.read_csv('data/ml-latest-small/movies.csv')
df_ratings = pd.read_csv('data/ml-latest-small/ratings.csv')

# Preprocesamiento de datos
def preprocesamiento():
    global df_movies, df_ratings

    df_movies = df_movies.drop_duplicates(subset=['movieId'])
    df_ratings = df_ratings.drop_duplicates(subset=['movieId', 'userId'])
    df_movies['content'] = df_movies['genres'].str.replace('|', ' ')
    df_movies['genres_set'] = df_movies['genres'].apply(lambda x: set(x.split('|')))
    df_ratings['timestamp'] = pd.to_datetime(df_ratings['timestamp'], unit='s')

preprocesamiento()

# Crear el DataFrame 'df_final'
df_final = pd.merge(df_ratings, df_movies, on='movieId')

# Función de recomendación de películas populares
def recomendaciones_populares():
    movie_counts = df_final['title'].value_counts()
    df_popular_movies = df_final[df_final['title'].isin(movie_counts[movie_counts > 210].index)]
    df_grouped = df_popular_movies.groupby('title').agg(mean_rating=('rating', 'mean'), vote_count=('rating', 'count'))
    df_movie_stats = df_grouped.reset_index()
    top_ten_movies = df_movie_stats.sort_values(by=['mean_rating', 'vote_count'], ascending=[False, False]).head(10)
    return top_ten_movies

# Función de recomendación ponderada
def recomendacion_ponderada():
    vote_counts = df_final['title'].value_counts()
    m = vote_counts.quantile(0.90)
    df_popular_movies = df_final[df_final['title'].isin(vote_counts[vote_counts >= m].index)]
    mean_ratings = df_popular_movies.groupby('title')['rating'].mean()

    df_movie_stats = pd.DataFrame({
        'mean_rating': mean_ratings,
        'vote_count': df_popular_movies.groupby('title')['rating'].count()
    })

    C = df_movie_stats['mean_rating'].mean()
    df_movie_stats['weighted_score'] = (df_movie_stats['vote_count'] / (df_movie_stats['vote_count'] + m) * df_movie_stats['mean_rating']) + (m / (m + df_movie_stats['vote_count']) * C)
    df_movie_stats = df_movie_stats.sort_values('weighted_score', ascending=False)

    return df_movie_stats.head(10)

# Función de Recomendación basada en Similitud de Jaccard
def recomendacion_jaccard(movie_title, n_recomendations=10):
    df_unique_movies = df_final.drop_duplicates(subset=['title'])
    input_movie_genres = df_unique_movies[df_unique_movies['title'] == movie_title]['genres_set'].values
    if len(input_movie_genres) == 0:
        return None
    input_movie_genres = input_movie_genres[0]

    similarities = {}
    for index, row in df_unique_movies.iterrows():
        title = row['title']
        genres = row['genres_set']
        if title != movie_title:
            similarity = jaccard_similarity(input_movie_genres, genres)
            similarities[title] = similarity

    df_similar_movies = pd.DataFrame(list(similarities.items()), columns=['title', 'jaccard_similarity'])
    df_similar_movies = df_similar_movies.sort_values(by='jaccard_similarity', ascending=False).head(n_recomendations)

    return df_similar_movies

# Función para calcular la Similitud de Jaccard
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Configuración de la matriz TF-IDF y la matriz de similitud de coseno
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df_movies['content'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Función de recomendación TF-IDF
def recomendacion_tf_idf(movie_id, n_recomendations=10):
    idx = df_movies[df_movies['movieId'] == movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recomendations+1]
    movie_indices = [i[0] for i in sim_scores]
    movie_scores = [i[1] for i in sim_scores]
    recomended_movies = df_movies.iloc[movie_indices].copy()
    recomended_movies['similarity_score'] = movie_scores
    return recomended_movies[['movieId', 'title', 'genres', 'similarity_score']]

# Configuración de la matriz de calificaciones para KNN
df_final = df_final.drop_duplicates(subset=['userId', 'title'])
ratings_matrix = df_final.pivot(index='userId', columns='title', values='rating')
avg_ratings = ratings_matrix.mean(axis=1)
ratings_matrix_normalized = ratings_matrix.sub(avg_ratings, axis=0).fillna(0)

# Entrenamiento del modelo KNN
knn_model = NearestNeighbors(metric='cosine', algorithm='auto')
knn_model.fit(ratings_matrix_normalized)

# Función de recomendación KNN
def recomendacion_knn(user_input, n_recomendations=10):
    if isinstance(user_input, int) and user_input in ratings_matrix.index:
        user_index = ratings_matrix.index.get_loc(user_input)
        distances, indices = knn_model.kneighbors(ratings_matrix_normalized.iloc[user_index, :].values.reshape(1, -1), n_neighbors=n_recomendations + 1)
    else:
        user_ratings = pd.Series(user_input).fillna(0).sub(avg_ratings.mean()).values.reshape(1, -1)
        distances, indices = knn_model.kneighbors(user_ratings, n_neighbors=n_recomendations + 1)

    distances, indices = distances.flatten()[1:], indices.flatten()[1:]
    similar_users_ratings = ratings_matrix_normalized.iloc[indices, :]
    weighted_ratings = similar_users_ratings.T.dot(1 - distances) / (1 - distances).sum()
    recommendations_df = pd.DataFrame(weighted_ratings, index=ratings_matrix.columns, columns=['weighted_score'])

    if isinstance(user_input, int):
        watched_movies = ratings_matrix.loc[user_input].dropna().index
        recommendations_df = recommendations_df[~recommendations_df.index.isin(watched_movies)]

    recommendations_df = recommendations_df.sort_values(by='weighted_score', ascending=False).head(n_recomendations)
    recommendations_df = recommendations_df.merge(df_movies[['title', 'genres']], left_index=True, right_on='title', how='left')

    return recommendations_df

# Función principal de Streamlit con Menú
def main():
    # Título de la aplicación
    st.title("Sistema de Recomendación de Películas")
    
    # Menú de navegación
    menu = ["Inicio", "Recomendaciones", "Análisis Exploratorio"]
    choice = st.sidebar.selectbox("Menú", menu)
    
    if choice == "Inicio":
        st.subheader("Inicio")
        st.write("Bienvenido al Sistema de Recomendación de Películas.")
        st.write("Navega a través del menú en la barra lateral para obtener recomendaciones personalizadas o explorar el análisis de datos.")
    
    elif choice == "Recomendaciones":
        st.subheader("Recomendaciones Personalizadas")
        
        # Selección de tipo de recomendación
        option = st.selectbox(
            "Elige el tipo de recomendación",
            ["Filtro Colaborativo (KNN)", "Similitud de Contenido (TF-IDF)", "Similitud de Jaccard", "Popular", "Weighted"]
        )
        
        num_recommendations = st.slider("Número de recomendaciones", 1, 20, 10)
        
        # Input del usuario para el ID
        user_id_input = st.text_input("Ingresa tu ID de usuario:", "")
        
        # Selección de película
        movie_title_input = st.selectbox("Selecciona una película (opcional):", ["Ninguna"] + list(df_movies['title'].unique()))
        
        if st.button("Obtener Recomendaciones"):
            try:
                recommendations = None
                
                if user_id_input.isdigit():
                    user_input = int(user_id_input)
                else:
                    user_input = None

                if movie_title_input != "Ninguna":
                    movie_title = movie_title_input
                else:
                    movie_title = None
                
                # Generar recomendaciones según la opción seleccionada y la entrada proporcionada
                if option == "Filtro Colaborativo (KNN)" and user_input is not None:
                    recommendations = recomendacion_knn(user_input, num_recommendations)
                elif option == "Similitud de Contenido (TF-IDF)" and movie_title:
                    movie_id = df_movies[df_movies['title'] == movie_title]['movieId'].values[0]
                    recommendations = recomendacion_tf_idf(movie_id, num_recommendations)
                elif option == "Similitud de Jaccard" and movie_title:
                    recommendations = recomendacion_jaccard(movie_title, num_recommendations)
                elif option == "Popular":
                    recommendations = recomendaciones_populares()
                elif option == "Weighted":
                    recommendations = recomendacion_ponderada()
                elif movie_title and not user_input:
                    # Si se selecciona una película y no se ingresa un ID de usuario, usar similitud de contenido
                    movie_id = df_movies[df_movies['title'] == movie_title]['movieId'].values[0]
                    recommendations = recomendacion_tf_idf(movie_id, num_recommendations)

                if recommendations is not None:
                    st.write("### Recomendaciones:")
                    st.dataframe(recommendations)
                else:
                    st.write("Por favor, selecciona una película o ingresa un ID de usuario válido para continuar.")
            except Exception as e:
                st.write(f"Error: {str(e)}")
    
    elif choice == "Análisis Exploratorio":
        st.subheader("Análisis Exploratorio de Datos (EDA)")

        # Distribución de Calificaciones
        st.write("### Distribución de Calificaciones")
        fig, ax = plt.subplots()
        sns.histplot(df_ratings['rating'], bins=10, kde=True, ax=ax)
        st.pyplot(fig)

        # Distribución de Calificaciones con Porcentajes
        st.write("### Distribución de Calificaciones con Porcentajes")
        fig, ax = plt.subplots()
        ax = sns.histplot(df_ratings['rating'], bins=10)
        total = len(df_ratings['rating'])
        for p in ax.patches:
            height = p.get_height()
            percentage = f'{height / total * 100:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = height
            ax.text(x, y, percentage, ha='center', va='bottom')
        st.pyplot(fig)

        # Boxplot de la distribución de calificaciones
        st.write("### Boxplot de la Distribución de Calificaciones")
        fig, ax = plt.subplots()
        sns.boxplot(x=df_ratings['rating'], ax=ax)
        st.pyplot(fig)

        # Cantidad de Calificaciones por Usuario
        ratings_per_user = df_ratings.groupby('userId')['rating'].count()
        st.write("### Cantidad de Calificaciones por Usuario")
        fig, ax = plt.subplots()
        sns.histplot(ratings_per_user, bins=50, kde=True, ax=ax)
        st.pyplot(fig)

        # Histograma de la Cantidad de Películas por Género
        genres_exploded = df_movies['genres'].str.split('|', expand=True).stack().reset_index(level=1, drop=True)
        st.write("### Histograma de la Cantidad de Películas por Género")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.countplot(y=genres_exploded, order=genres_exploded.value_counts().index, palette='viridis', ax=ax)
        st.pyplot(fig)

        # Cantidad de Calificaciones por Película
        ratings_per_movie = df_ratings.groupby('movieId')['rating'].count()
        st.write("### Cantidad de Calificaciones por Película")
        fig, ax = plt.subplots()
        sns.histplot(ratings_per_movie, bins=50, kde=True, ax=ax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
