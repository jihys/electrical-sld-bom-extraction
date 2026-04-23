// ──────────────────────────────────────────────────────────────────
// Service Bus – Queue for async tasks, Topic for progress events
// ──────────────────────────────────────────────────────────────────

@description('Service Bus namespace name')
param namespaceName string

@description('Azure region')
param location string

@description('Resource tags')
param tags object

// ── Namespace ─────────────────────────────────────────────────────
resource sbNamespace 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' = {
  name: namespaceName
  location: location
  tags: tags
  sku: {
    name: 'Standard'  // Standard required for Topics
    tier: 'Standard'
  }
}

// ── Queue: pipeline-tasks ─────────────────────────────────────────
resource pipelineTasksQueue 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = {
  parent: sbNamespace
  name: 'pipeline-tasks'
  properties: {
    lockDuration: 'PT5M'          // 5 min lock for long-running tasks
    maxSizeInMegabytes: 1024
    requiresDuplicateDetection: true
    duplicateDetectionHistoryTimeWindow: 'PT10M'
    maxDeliveryCount: 3
    defaultMessageTimeToLive: 'P1D'  // 1 day TTL
    deadLetteringOnMessageExpiration: true
  }
}

// ── Topic: pipeline-events ────────────────────────────────────────
resource pipelineEventsTopic 'Microsoft.ServiceBus/namespaces/topics@2022-10-01-preview' = {
  parent: sbNamespace
  name: 'pipeline-events'
  properties: {
    maxSizeInMegabytes: 1024
    defaultMessageTimeToLive: 'PT1H'  // 1 hour TTL for events
    supportOrdering: true
  }
}

// ── Subscription: ui-updates ──────────────────────────────────────
resource uiUpdatesSub 'Microsoft.ServiceBus/namespaces/topics/subscriptions@2022-10-01-preview' = {
  parent: pipelineEventsTopic
  name: 'ui-updates'
  properties: {
    lockDuration: 'PT30S'
    maxDeliveryCount: 5
    defaultMessageTimeToLive: 'PT1H'
    deadLetteringOnMessageExpiration: false
    autoDeleteOnIdle: 'P1D'
  }
}

// ── Subscription: api-updates ─────────────────────────────────────
resource apiUpdatesSub 'Microsoft.ServiceBus/namespaces/topics/subscriptions@2022-10-01-preview' = {
  parent: pipelineEventsTopic
  name: 'api-updates'
  properties: {
    lockDuration: 'PT30S'
    maxDeliveryCount: 5
    defaultMessageTimeToLive: 'PT1H'
    deadLetteringOnMessageExpiration: false
    autoDeleteOnIdle: 'P1D'
  }
}

// ── Outputs ───────────────────────────────────────────────────────
output namespaceName string = sbNamespace.name
output endpoint string = '${sbNamespace.name}.servicebus.windows.net'
output pipelineTasksQueue string = pipelineTasksQueue.name
output pipelineEventsTopic string = pipelineEventsTopic.name
