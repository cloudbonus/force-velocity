FROM rabbitmq:4.0.3-management

RUN rabbitmq-plugins enable --offline rabbitmq_management

EXPOSE 5672 15672

ENV RABBITMQ_DEFAULT_USER=user
ENV RABBITMQ_DEFAULT_PASS=password

CMD ["rabbitmq-server"]