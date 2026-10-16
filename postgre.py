"""
Pipeline ETL Automatizado (Dual Load: AWS S3 + PostgreSQL)
-----------------------------------------------------------
Descripción:
    Extrae un dataset transaccional, aplica transformaciones
    (discretización binning, normalización MinMaxScaler y feature engineering)
    y realiza una carga dual:
      1. Formato Parquet binario comprimido en AWS S3 (Data Lake).
      2. Tabla relacional en base de datos PostgreSQL local (Data Warehouse).

Autor: Equipo de Ingeniería de Datos
Entorno: Python 3.14 (venv: proyecto001)
"""

import io
import sys
import logging
import boto3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import yaml

# Configuración del sistema de logs para trazabilidad
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def cargar_configuracion(ruta_config: str = "config.yaml") -> dict:
    """Carga los parámetros y credenciales del archivo YAML.

    Args:
        ruta_config (str): Ruta relativa del archivo de configuración.

    Returns:
        dict: Diccionario con la estructura de configuración.
    """
    try:
        with open(ruta_config, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            logging.info("Configuración cargada exitosamente desde %s", ruta_config)
            return config
    except FileNotFoundError:
        logging.error("El archivo %s no fue encontrado.", ruta_config)
        sys.exit(1)
    except yaml.YAMLError as err:
        logging.error("Error al parsear el archivo YAML: %s", err)
        sys.exit(1)


def extraer_datos() -> pd.DataFrame:
    """Fase 1: Extracción de datos (Extract).

    Simula la ingesta de un dataset crudo desde una fuente transaccional.

    Returns:
        pd.DataFrame: DataFrame inicial con datos sin procesar.
    """
    logging.info("=== 1. FASE DE EXTRACCIÓN (Extract) ===")
    raw_data = {
        "cliente": ["A", "B", "C", "D"],
        "edad": [20, 40, 60, 30],
        "monto_compras": [1200.0, 50000.0, 30000.0, 15000.0],
    }
    df = pd.DataFrame(raw_data)
    logging.info("Datos extraídos correctamente (%d registros).", len(df))
    return df


def transformar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Fase 2: Transformación de datos (Transform).

    Aplica generalización (binning), normalización (MinMaxScaler)
    y derivación de nuevas métricas (Feature Engineering).

    Args:
        df (pd.DataFrame): DataFrame crudo.

    Returns:
        pd.DataFrame: DataFrame transformado y enriquecido.
    """
    logging.info("=== 2. FASE DE TRANSFORMACIÓN (Transform) ===")
    
    # 2.1. Discretización de Edad (Binning)
    df["categoria_edad"] = pd.cut(
        df["edad"],
        bins=[0, 25, 50, 100],
        labels=["Joven", "Adulto", "Senior"],
    )

    # 2.2. Normalización de Monto de Compras (MinMaxScaler [0, 1])
    scaler = MinMaxScaler()
    df["monto_normalizado"] = scaler.fit_transform(df[["monto_compras"]])

    # 2.3. Feature Engineering: Score Promocional
    df["score_promocional"] = df["monto_normalizado"] * 100

    logging.info("Dataset transformado exitosamente:\n%s", df.to_string())
    return df


def cargar_a_s3(df: pd.DataFrame, config_aws: dict) -> None:
    """Subfase 3.1: Carga a AWS S3 en formato Parquet (Data Lake).

    Serializa el DataFrame a un buffer en memoria RAM en formato Parquet
    y lo transmite mediante Boto3 hacia el bucket configurado.

    Args:
        df (pd.DataFrame): Dataset procesado.
        config_aws (dict): Parámetros de conexión y credenciales para AWS.
    """
    logging.info("Iniciando carga a AWS S3...")
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=config_aws["aws_access_key_id"],
            aws_secret_access_key=config_aws["aws_secret_access_key"],
            aws_session_token=config_aws["aws_session_token"],
            region_name=config_aws["region_name"],
        )

        # Serialización en memoria en formato Parquet
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        bucket_name = config_aws["bucket_name"]
        s3_key = "processed/etl_output.parquet"

        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=buffer.getvalue(),
        )
        logging.info("-> Éxito: Archivo Parquet cargado en S3 (%s/%s).", bucket_name, s3_key)

    except Exception as err:
        logging.error("Fallo durante la carga a AWS S3: %s", err)
        raise err


def cargar_a_postgresql(df: pd.DataFrame, config_pg: dict) -> None:
    """Subfase 3.2: Carga a PostgreSQL local (Data Warehouse).

    Crea un motor de conexión SQLAlchemy e inserta/reemplaza
    la tabla objetivo en la base de datos relacional.

    Args:
        df (pd.DataFrame): Dataset procesado.
        config_pg (dict): Parámetros de conexión para PostgreSQL.
    """
    logging.info("Iniciando carga a PostgreSQL local...")
    try:
        user = config_pg["user"]
        password = config_pg["password"]
        host = config_pg["host"]
        port = config_pg["port"]
        database = config_pg["database"]

        # Creación del engine SQLAlchemy
        connection_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(connection_url)

        # Inserción idempotente (reemplaza si existe)
        table_name = "clientes_transformados"
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists="replace",
            index=False,
        )
        logging.info("-> Éxito: Tabla '%s' guardada en PostgreSQL local (%s).", table_name, database)

    except Exception as err:
        logging.error("Fallo durante la carga a PostgreSQL: %s", err)
        raise err


def pipeline_etl_completo():
    """Función principal que coordina la ejecución secuencial del Pipeline ETL."""
    logging.info("==================================================")
    logging.info("INICIANDO EJECUCIÓN DEL PIPELINE ETL AUTOMATIZADO")
    logging.info("==================================================")

    # Cargar archivo de parámetros
    config = cargar_configuracion("config.yaml")

    # 1. Extracción
    df_raw = extraer_datos()

    # 2. Transformación
    df_transformed = transformar_datos(df_raw)

    # 3. Carga Dual
    logging.info("=== 3. FASE DE CARGA (Load) ===")
    cargar_a_s3(df_transformed, config["aws"])
    cargar_a_postgresql(df_transformed, config["postgres"])

    logging.info("==================================================")
    logging.info("PIPELINE ETL FINALIZADO CON ÉXITO")
    logging.info("==================================================")


if __name__ == "__main__":
    pipeline_etl_completo()