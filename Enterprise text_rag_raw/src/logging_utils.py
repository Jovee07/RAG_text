import logging
import sys

def setup_logging():
    logger = logging.getLogger("rag_service")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter= logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger

#Logging to AWS CloudWatch

# import logging
# import sys
# import watchtower  # pip install watchtower
# import boto3

# def setup_logging():
#     logger = logging.getLogger("rag_service")
#     logger.setLevel(logging.INFO)

#     # Console handler (stdout)
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_formatter = logging.Formatter(
#         fmt="%(asctime)s %(levelname)s %(name)s %(message)s"
#     )
#     console_handler.setFormatter(console_formatter)
#     logger.addHandler(console_handler)

#     # CloudWatch handler
#     session = boto3.Session(region_name="us-east-1")  # adjust region
#     cloudwatch_handler = watchtower.CloudWatchLogHandler(
#         boto3_session=session,
#         log_group="rag_service_logs",   # CloudWatch log group
#         stream_name="rag_stream"        # CloudWatch log stream
#     )
#     cloudwatch_handler.setFormatter(console_formatter)
#     logger.addHandler(cloudwatch_handler)

#     return logger

# # Usage
# logger = setup_logging()
# logger.info("RAG system initialized")
# logger.error("Failed to retrieve documents")



#Logging to Azure Application Insights
# import logging
# import sys
# from opencensus.ext.azure.log_exporter import AzureLogHandler

# def setup_logging():
#     logger = logging.getLogger("rag_service")
#     logger.setLevel(logging.INFO)

#     # Console handler
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_formatter = logging.Formatter(
#         fmt="%(asctime)s %(levelname)s %(name)s %(message)s"
#     )
#     console_handler.setFormatter(console_formatter)
#     logger.addHandler(console_handler)

#     # Azure Application Insights handler
#     connection_string = "InstrumentationKey=YOUR_INSTRUMENTATION_KEY"
#     azure_handler = AzureLogHandler(connection_string=connection_string)
#     logger.addHandler(azure_handler)

#     return logger

# # Usage
# logger = setup_logging()
# logger.info("RAG system initialized")
# logger.warning("Embedding dimension mismatch detected")
