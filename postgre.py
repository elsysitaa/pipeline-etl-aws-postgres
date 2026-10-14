import io
import boto3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import yaml


def cargar_configuracion():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


def pipeline_etl_completo():
    # Cargar variables desde el YAML
    config = cargar_configuracion()

    print("=== 1. FASE DE EXTRACCIÓN (Extract) ===")
    raw_data = {
        "cliente": ["A", "B", "C", "D"],
        "edad": [20, 40, 60, 30],
        "monto_compras": [1200.0, 50000.0, 30000.0, 15000.0],
    }
    df_raw = pd.DataFrame(raw_data)

    print("=== 2. FASE DE TRANSFORMACIÓN (Transform) ===")
    df_raw["categoria_edad"] = pd.cut(
        df_raw["edad"],
        bins=[0, 25, 50, 100],
        labels=["Joven", "Adulto", "Senior"],
    )

    scaler = MinMaxScaler()
    df_raw["monto_normalizado"] = scaler.fit_transform(
        df_raw[["monto_compras"]]
    )
    df_raw["score_promocional"] = df_raw["monto_normalizado"] * 100

    print("Dataset Transformado:")
    print(df_raw)

    print("\n=== 3. FASE DE CARGA (Load) ===")

    # -------------------------------------------------------------
    # 3.1. CARGA A AWS S3
    # -------------------------------------------------------------
    aws_cfg = config["aws"]
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_cfg["aws_access_key_id"],
        aws_secret_access_key=aws_cfg["aws_secret_access_key"],
        aws_session_token=aws_cfg["aws_session_token"],
        region_name=aws_cfg["region_name"],
    )

    buffer = io.BytesIO()
    df_raw.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3_client.put_object(
        Bucket=aws_cfg["bucket_name"],
        Key="processed/etl_output.parquet",
        Body=buffer.getvalue(),
    )
    print(
        f"-> Éxito: Archivo Parquet cargado en S3 ({aws_cfg['bucket_name']}/processed/)."
    )

    # -------------------------------------------------------------
    # 3.2. CARGA A POSTGRESQL LOCAL
    # -------------------------------------------------------------
    pg_cfg = config["postgres"]
    engine = create_engine(
        f"postgresql://{pg_cfg['user']}:{pg_cfg['password']}@{pg_cfg['host']}:{pg_cfg['port']}/{pg_cfg['database']}"
    )

    df_raw.to_sql(
        name="clientes_transformados",
        con=engine,
        if_exists="replace",
        index=False,
    )
    print(
        "-> Éxito: Tabla 'clientes_transformados' guardada en PostgreSQL local."
    )


if __name__ == "__main__":
    pipeline_etl_completo()