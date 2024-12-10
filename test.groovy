stages {
    stage('TRDA Health Checks') {
        steps {
            catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                script {
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
                            def output = ""
                            def exitCode = sh(script: """
                                ansible-playbook trda-health.yaml -e application=${app} --tags nft 2>&1
                            """, returnStatus: true, returnStdout: true).trim()

                            echo "Exit Code for ${app}: ${exitCode}"
                            if (exitCode != 0) {
                                echo "Ansible playbook failed for ${app}"
                                stageFailed = true
                            }

                            if (output?.trim()) {
                                output.eachLine { line ->
                                    if (line.contains("failed")) {
                                        echo "Failed Line Detected: ${line}"
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
                            } else {
                                echo "No output for ${app}. Skipping processing."
                            }

                            if (failedUrls.isEmpty()) {
                                echo "Application ${app} passed the health check."
                            }

                        } catch (Exception e) {
                            echo "Error processing application ${app}: ${e.message}"
                            stageFailed = true
                        }
                    }

                    // Handle stage failure and email notifications
                    if (stageFailed) {
                        failedUrls.each { failedUrl ->
                            echo "Sending email for failed URL: ${failedUrl}"
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
