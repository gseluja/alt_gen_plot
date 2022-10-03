FROM python:3.8.12-buster
COPY alt_plot_gen /alt_plot_gen
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn alt_plot_gen.api.fast:app --host 0.0.0.0
