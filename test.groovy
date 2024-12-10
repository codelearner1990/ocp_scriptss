pipeline {
    agent any

    stages {
        stage('TRDA Health Checks') {
            steps {
                catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                    script {
                        def teamsYaml = readYaml file: 'teams.yaml'
                        echo "Parsed YAML: ${teamsYaml}"

                        def applications = teamsYaml.team_mappings.trda.keySet() ?: []
                        if (applications.isEmpty()) {
                            error "No applications found for TRDA in team mappings"
                        }

                        def failedUrls = []
                        def stageFailed = false

                        echo "TRDA applications: ${applications}"

                        applications.each { app ->
                            echo "Processing application: ${app}"
                            def teamOwner = teamsYaml.team_mappings.trda[app]
                            if (!teamOwner) {
                                echo "No team owner found for application: ${app}. Skipping email."
                                return
                            }

                            try {
                                def output = new StringWriter()
                                def error = new StringWriter()
                                def process = "ansible-playbook trda-health.yaml -e application=${app} --tags nft".execute()
                                process.consumeProcessOutput(output, error)
                                process.waitFor()
                                def exitCode = process.exitValue()

                                echo "Exit Code for ${app}: ${exitCode}"
                                echo "Playbook Output for ${app}:\n${output.toString()}"
                                echo "Error Output for ${app}:\n${error.toString()}"

                                // Process output to extract failed URLs
                                output.toString().eachLine { line ->
                                    if (line.contains("failed")) {
                                        echo "Failed Line: ${line}"
                                        if (line.contains('"url":')) {
                                            def urlStart = line.indexOf('"url":') + 7
                                            def urlEnd = line.indexOf('"', urlStart)
                                            if (urlEnd > urlStart) {
                                                def failedUrl = line.substring(urlStart, urlEnd).trim()
                                                echo "Captured failed URL: ${failedUrl}"
                                                failedUrls.add(failedUrl)
                                            }
                                        }
                                    }
                                }

                                if (!failedUrls.isEmpty()) {
                                    echo "Failed URLs for ${app}: ${failedUrls}"
                                    stageFailed = true
                                } else {
                                    echo "Application ${app} passed the health check."
                                }
                            } catch (Exception e) {
                                echo "Error processing application ${app}: ${e.message}"
                                stageFailed = true
                            }
                        }

                        // Send emails for failures if any
                        if (stageFailed) {
                            failedUrls.each { failedUrl ->
                                echo "Sending email for failed application: ${failedUrl}"
                                sh """
                                    ansible-playbook notify_email.yaml -e product_family=trda \
                                        -e applications=${app} -e team_owner=${teamOwner} -e failed_urls=${failedUrl}
                                """
                            }
                        } else {
                            echo "All applications passed health checks."
                        }
                    }
                }
            }
        }
    }
}
