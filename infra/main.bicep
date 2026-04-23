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

@description('App Service SKU')
param appServiceSku string = 'B2'

@description('Azure OpenAI endpoint (passed to App Service)')
param azureOpenAiEndpoint string = ''

@description('Azure OpenAI deployment name')
param azureOpenAiDeployment string = 'gpt-5.4'

@description('Azure OpenAI API version')
param azureOpenAiApiVersion string = '2025-03-01-preview'

@description('Azure Document Intelligence endpoint')
param azureDiEndpoint string = ''

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

// ── App Service (Streamlit UI) ────────────────────────────────────
module appService 'modules/appservice.bicep' = {
  name: 'appservice-${baseName}'
  params: {
    appName: 'app-${baseName}-${environment}'
    location: location
    tags: tags
    skuName: appServiceSku
    cosmosEndpoint: cosmosDb.outputs.endpoint
    cosmosDatabase: cosmosDb.outputs.databaseName
    storageBlobEndpoint: storage.outputs.blobEndpoint
    storageAccountName: storage.outputs.storageAccountName
    serviceBusEndpoint: serviceBus.outputs.endpoint
    azureOpenAiEndpoint: azureOpenAiEndpoint
    azureOpenAiDeployment: azureOpenAiDeployment
    azureOpenAiApiVersion: azureOpenAiApiVersion
    azureDiEndpoint: azureDiEndpoint
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

output appServiceName string = appService.outputs.appServiceName
output appServiceUrl string = appService.outputs.appServiceUrl
output appServicePrincipalId string = appService.outputs.principalId
