#!/bin/bash

# Variables
DEPLOYMENT_NAME="<deployment-name>"
NAMESPACE="<namespace>"
DESIRED_REPLICAS=<desired-replicas>
WAIT_TIME=300  # 300 seconds = 5 minutes
LOG_FILE="deployment_scale.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}

# Function to list pods
list_pods() {
    log_message "Listing pods in namespace $NAMESPACE:"
    oc get pods -n $NAMESPACE 2>&1 | tee -a $LOG_FILE
}

# Start logging
log_message "Script execution started."

# List pods before scaling down
list_pods

# Scale down the deployment
log_message "Scaling down deployment $DEPLOYMENT_NAME to 0 replicas."
oc scale deployment $DEPLOYMENT_NAME --replicas=0 -n $NAMESPACE 2>&1 | tee -a $LOG_FILE

# Wait for the specified period
log_message "Waiting for $WAIT_TIME seconds."
sleep $WAIT_TIME

# List pods after scaling down
list_pods

# Scale up the deployment
log_message "Scaling up deployment $DEPLOYMENT_NAME to $DESIRED_REPLICAS replicas."
oc scale deployment $DEPLOYMENT_NAME --replicas=$DESIRED_REPLICAS -n $NAMESPACE 2>&1 | tee -a $LOG_FILE

# List pods after scaling up
list_pods

# End logging
log_message "Script execution completed."
