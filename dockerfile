FROM openjdk:17

# Create a new group and user with desired user and group IDs
RUN groupadd -g 1001 appgroup && useradd -u 1001 -g appgroup -ms /bin/bash appuser

# Copy Files
WORKDIR /usr/src/app
RUN chown -R appuser:appgroup /usr/src/app
COPY --chown=appuser:appgroup . .

# Switch to the appuser
USER appuser

# Install
RUN ./mvnw -Dmaven.test.skip=true package

# Docker Run Command
EXPOSE 8080
CMD ["java","-jar","/usr/src/app/target/image-object-detection-1.0.0-SNAPSHOT.jar"]
