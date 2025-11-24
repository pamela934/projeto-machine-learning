from flask import Flask, request, jsonify
import joblib
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__)

modelo = joblib.load("modelo_categoria.pkl")
encoders = joblib.load("encoders.pkl")
hospitais = pd.read_csv("hospitais_final.csv")

def distancia_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    sintoma = data.get("sintoma")
    intensidade = data.get("intensidade")
    urgencia = data.get("urgencia")
    lat = data.get("latitude")
    lon = data.get("longitude")

    if not all([sintoma, intensidade, urgencia, lat, lon]):
        return jsonify({"erro": "sintoma, intensidade, urgencia, latitude e longitude são obrigatórios"}), 400

    X = {}
    for col, val in [("sintoma", sintoma), ("intensidade", intensidade), ("urgencia", urgencia)]:
        X[col] = encoders[col].transform([val])[0]

    X_df = pd.DataFrame([X])

    cat_num = modelo.predict(X_df)[0]
    categoria_pred = encoders["categoria"].inverse_transform([cat_num])[0]

    categorias_finais = [categoria_pred]

    if urgencia.lower() == "baixa" and categoria_pred == "SPA":
        categorias_finais.append("GERAL")

    df_filtrado = hospitais[hospitais["categoria"].isin(categorias_finais)].copy()

    df_filtrado["dist_km"] = df_filtrado.apply(
        lambda row: distancia_km(lat, lon, row["lat"], row["lon"]),
        axis=1
    )

    df_ordenado = df_filtrado.sort_values("dist_km").reset_index(drop=True)

    vistos = set()
    recomendados = []

    for _, row in df_ordenado.iterrows():
        if row["categoria"] not in vistos:
            recomendados.append({
                "nome": row["nome"],
                "categoria": row["categoria"],
                "lat": row["lat"],
                "lon": row["lon"],
                "dist_km": round(row["dist_km"], 2)
            })
            vistos.add(row["categoria"])
        if len(recomendados) >= 2:
            break

    return jsonify({
        "categoria_predita": categoria_pred,
        "categorias_finais": categorias_finais,
        "hospitais_recomendados": recomendados
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
