// ──────────────────────────────────────────────────────────────────
// Main Bicep template – SLD BOM Extraction Infrastructure
// Deploys: Cosmos DB (Serverless), Storage Account, Service Bus
// ──────────────────────────────────────────────────────────────────

targetScope = 'resourceGroup'

@description('Base name prefix for all resources')
param baseName string

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Environment tag (dev / staging / prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

@description('Tags applied to every resource')
param tags object = {
  project: 'sld-bom-extraction'
  environment: environment
}

// ── Cosmos DB ─────────────────────────────────────────────────────
module cosmosDb 'modules/cosmosdb.bicep' = {
  name: 'cosmosdb-${baseName}'
  params: {
    accountName: 'cosmos-${baseName}-${environment}'
    location: location
    tags: tags
  }
}

// ── Storage Account ───────────────────────────────────────────────
module storage 'modules/storage.bicep' = {
  name: 'storage-${baseName}'
  params: {
    storageAccountName: replace('st${baseName}${environment}', '-', '')
    location: location
    tags: tags
  }
}

// ── Service Bus ───────────────────────────────────────────────────
module serviceBus 'modules/servicebus.bicep' = {
  name: 'servicebus-${baseName}'
  params: {
    namespaceName: 'sb-${baseName}-${environment}'
    location: location
    tags: tags
  }
}

// ── Outputs ───────────────────────────────────────────────────────
output cosmosDbEndpoint string = cosmosDb.outputs.endpoint
output cosmosDbAccountName string = cosmosDb.outputs.accountName
output cosmosDbDatabaseName string = cosmosDb.outputs.databaseName

output storageAccountName string = storage.outputs.storageAccountName
output storageBlobEndpoint string = storage.outputs.blobEndpoint
output pipelineArtifactsContainer string = storage.outputs.pipelineArtifactsContainer
output fileCacheContainer string = storage.outputs.fileCacheContainer

output serviceBusNamespace string = serviceBus.outputs.namespaceName
output serviceBusEndpoint string = serviceBus.outputs.endpoint
output pipelineTasksQueue string = serviceBus.outputs.pipelineTasksQueue
output pipelineEventsTopic string = serviceBus.outputs.pipelineEventsTopic
