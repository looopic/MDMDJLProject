FROM openjdk:17-jdk-slim
# Copy Files
WORKDIR /usr/src/app
COPY . .
# Install
RUN ./mvnw org.springframework.boot:spring-boot-maven-plugin:run
# Docker Run Command
EXPOSE 8080
CMD ["java","-jar","/usr/src/app/target/playground-0.0.1-SNAPSHOT.jar"]