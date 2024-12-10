pipeline {
    agent any

    stages {
        stage('TRDA Health Checks') {
            steps {
                catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                    script {
                        def failedUrls = []
                        def stageFailed = false

                        echo "Starting TRDA Health Checks"

                        applications.each { app ->
                            echo "Processing application: ${app}"
                            def teamOwner = teamsYaml.team_mappings.trda[app]
                            if (!teamOwner) {
                                echo "No team owner found for application: ${app}. Skipping email."
                                return
                            }

                            try {
                                // Ensure the workspace directory exists
                                sh "mkdir -p ${env.WORKSPACE}"

                                // Define and reset the output file
                                def outputFile = "${env.WORKSPACE}/playbook_output_${app}.txt"
                                sh "echo '' > ${outputFile}"

                                // Run the ansible playbook and capture output
                                def playbookOutput = sh(
                                    script: """
                                        ansible-playbook trda-health.yaml -e application=${app} --tags nft > ${outputFile} 2>&1 || true
                                    """,
                                    returnStdout: true
                                ).trim()

                                echo "Playbook Output for ${app} saved to ${outputFile}"
                                echo "Raw Output:\n${playbookOutput}"

                                // Parse output for failed URLs
                                playbookOutput.eachLine { line ->
                                    try {
                                        if (line?.trim() && line.contains("failed") && line.contains('"url":')) {
                                            echo "Processing failed line: ${line}"
                                            def urlStart = line.indexOf('"url":') + 7
                                            def urlEnd = line.indexOf('"', urlStart)
                                            if (urlEnd > urlStart) {
                                                def failedUrl = line.substring(urlStart, urlEnd).trim()
                                                echo "Captured failed URL: ${failedUrl}"
                                                failedUrls.add([app: app, url: failedUrl, teamOwner: teamOwner])
                                            }
                                        }
                                    } catch (Exception e) {
                                        echo "Error processing line: ${line}. Error: ${e.message}"
                                    }
                                }

                                if (failedUrls.isEmpty()) {
                                    echo "Application ${app} passed the health check."
                                } else {
                                    echo "Application ${app} has failed URLs: ${failedUrls}"
                                    stageFailed = true
                                }
                            } catch (Exception e) {
                                echo "Error processing application ${app}: ${e.message}"
                                stageFailed = true
                            }
                        }

                        // Send emails for failed URLs
                        if (failedUrls) {
                            failedUrls.each { failed ->
                                echo "Sending email for failed URL: ${failed.url}"
                                sh """
                                    ansible-playbook notify_email.yaml \
                                        -e product_family=trda \
                                        -e applications=${failed.app} \
                                        -e team_owner=${failed.teamOwner} \
                                        -e failed_url=${failed.url}
                                """
                            }
                        } else {
                            echo "No failed URLs found. All applications passed health checks."
                        }

                        // Mark the stage as failed if any failures occurred
                        if (stageFailed) {
                            error "One or more applications failed health checks."
                        }
                    }
                }
            }
        }
    }
}
