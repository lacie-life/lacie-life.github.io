---
title: Amazon DocumentDB
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-11-19 11:11:14 +0700
categories: [Devops]
tags: [Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# DocumentDB

## 1. What Is Amazon DocumentDB (with MongoDB Compatibility)?

Amazon DocumentDB (with MongoDB compatibility) is a fast, reliable, and fully managed database service. Amazon DocumentDB makes it easy to set up, operate, and scale MongoDB-compatible databases in the cloud. With Amazon DocumentDB, you can run the same application code and use the same drivers and tools that you use with MongoDB.

## Overview of Amazon DocumentDB

The following are some high-level features of Amazon DocumentDB:

- Amazon DocumentDB automatically grows the size of your storage volume as your database storage needs grow. Your storage volume grows in increments of 10 GB, up to a maximum of 64 TB. You don't need to provision any excess storage for your cluster to handle future growth.
- With Amazon DocumentDB, you can increase read throughput to support high-volume application requests by creating up to 15 replica instances. Amazon DocumentDB replicas share the same underlying storage, lowering costs and avoiding the need to perform writes at the replica nodes. This capability frees up more processing power to serve read requests and reduces the replica lag time—often down to single digit milliseconds. You can add replicas in minutes regardless of the storage volume size. Amazon DocumentDB also provides a reader endpoint, so the application can connect without having to track replicas as they are added and removed.
    
- Amazon DocumentDB lets you scale the compute and memory resources for each of your instances up or down. Compute scaling operations typically complete in a few minutes.
- Amazon DocumentDB runs in Amazon Virtual Private Cloud (Amazon VPC), so you can isolate your database in your own virtual network. You can also configure firewall settings to control network access to your cluster.
- Amazon DocumentDB continuously monitors the health of your cluster. On an instance failure, Amazon DocumentDB automatically restarts the instance and associated processes. Amazon DocumentDB doesn't require a crash recovery replay of database redo logs, which greatly reduces restart times. Amazon DocumentDB also isolates the database cache from the database process, enabling the cache to survive an instance restart.
- On instance failure, Amazon DocumentDB automates failover to one of up to 15 Amazon DocumentDB replicas that you create in other Availability Zones. If no replicas have been provisioned and a failure occurs, Amazon DocumentDB tries to create a new Amazon DocumentDB instance automatically.
- The backup capability in Amazon DocumentDB enables point-in-time recovery for your cluster. This feature allows you to restore your cluster to any second during your retention period, up to the last 5 minutes. You can configure your automatic backup retention period up to 35 days. Automated backups are stored in Amazon Simple Storage Service (Amazon S3), which is designed for 99.999999999% durability (really???????). Amazon DocumentDB backups are automatic, incremental, and continuous, and they have no impact on your cluster performance.
- With Amazon DocumentDB, you can encrypt your databases using keys that you create and control through AWS Key Management Service (AWS KMS). On a database cluster running with Amazon DocumentDB encryption, data stored at rest in the underlying storage is encrypted. The automated backups, snapshots, and replicas in the same cluster are also encrypted.
    
    (More details: https://aws.amazon.com/products/databases/)

## Clusters

A cluster consists of 0 to 16 instances and a cluster storage volume that manages the data for those instances. All writes are done through the primary instance. All instances (primary and replicas) support reads. The cluster's data is stored in the cluster volume with copies in three different Availability Zones.

   ![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/how-it-works-01c.png?raw=true)

## Instances

An Amazon DocumentDB instance is an isolated database environment in the cloud. An instance can contain multiple user-created databases. You can create and modify an instance using the AWS Management Console or the AWS CLI.

The computation and memory capacity of an instance is determined by its instance class. You can select the instance that best meets your needs. If your needs change over time, you can choose a different instance class.

Amazon DocumentDB instances run only in the Amazon VPC environment. Amazon VPC gives you control of your virtual networking environment: You can choose your own IP address range, create subnets, and configure routing and access control lists (ACLs).

Before you can create Amazon DocumentDB instances, you must create a cluster to contain the instances.

Not all instance classes are supported in every region. 
 ## Regions and Availability Zones

Regions and Availability Zones define the physical locations of your cluster and instances.

### Regions

AWS Cloud computing resources are housed in highly available data center facilities in different areas of the world (for example, North America, Europe, or Asia). Each data center location is called a Region.

Each AWS Region is designed to be completely isolated from the other AWS Regions. Within each are multiple Availability Zones. By launching your nodes in different Availability Zones, you can achieve the greatest possible fault tolerance. The following diagram shows a high-level view of how AWS Regions and Availability Zones work.

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/docdb-regions-and-azs.png?raw=true)

### Availability Zones

Each AWS Region contains multiple distinct locations called Availability Zones. Each Availability Zone is engineered to be isolated from failures in other Availability Zones, and to provide inexpensive, low-latency network connectivity to other Availability Zones in the same Region. By launching instances for a given cluster in multiple Availability Zones, you can protect your applications from the unlikely event of an Availability Zone failing.

The Amazon DocumentDB architecture separates storage and compute. For the storage layer, Amazon DocumentDB replicates six copies of your data across three AWS Availability Zones. As an example, if you are launching an Amazon DocumentDB cluster in a Region that only supports two Availability Zones, your data storage will be replicated six ways across three Availability Zones but your compute instances will only be available in two Availability Zones.

The following table lists the number of Availability Zones that you can use in a given AWS Region to provision compute instances for your cluster.

#### Amazon DocumentDB Pricing

Amazon DocumentDB clusters are billed based on the following components. Amazon DocumentDB does not currently have a free tier so creating a cluster will incur costs.

- Instance hours (per hour) —Based on the instance class of the instance (for example, db.r5.xlarge). Pricing is listed on a per-hour basis, but bills are calculated down to the second and show times in decimal form. Amazon DocumentDB usage is billed in one second increments, with a minimum of 10 minutes. For more information, see Managing Instance Classes.

- I/O requests (per 1 million requests per month) — Total number of storage I/O requests that you make in a billing cycle.

- Backup storage (per GiB per month) — Backup storage is the storage that is associated with automated database backups and any active database snapshots that you have taken. Increasing your backup retention period or taking additional database snapshots increases the backup storage consumed by your database. Backup storage is metered in GB-months and per second does not apply. For more information, see Backing Up and Restoring in Amazon DocumentDB.

- Data transfer (per GB) — Data transfer in and out of your instance from or to the internet or other AWS Regions.

#### Monitoring

There are several ways that you can track the performance and health of an instance. You can use the free Amazon CloudWatch service to monitor the performance and health of an instance. You can find performance charts on the Amazon DocumentDB console. You can subscribe to Amazon DocumentDB events to be notified when changes occur with an instance, snapshot, parameter group, or security group.

#### Interfaces

There are multiple ways for you to interact with Amazon DocumentDB, including the AWS Management Console and the AWS CLI.

### AWS Management Console

The AWS Management Console is a simple web-based user interface. You can manage your clusters and instances from the console with no programming required. To access the Amazon DocumentDB console, sign in to the AWS Management Console and open the Amazon DocumentDB console at https://console.aws.amazon.com/docdb.

### AWS CLI

You can use the AWS Command Line Interface (AWS CLI) to manage your Amazon DocumentDB clusters and instances. With minimal configuration, you can start using all of the functionality provided by the Amazon DocumentDB console from your favorite terminal program.

### The mongo Shell

To connect to your cluster to create, read, update, delete documents in your databases, you can use the mongo shell with Amazon DocumentDB. To download and install the mongo 3.6 shell, see Step 3: Access and Use Your Amazon DocumentDB Cluster Using the mongo Shell.
### MongoDB Drivers

For developing and writing applications against an Amazon DocumentDB cluster, you can also use the MongoDB drivers with Amazon DocumentDB.

## 2. How It Works?

Amazon DocumentDB (with MongoDB compatibility) is a fully managed, MongoDB-compatible database service. With Amazon DocumentDB, you can run the same application code and use the same drivers and tools that you use with MongoDB. Amazon DocumentDB is compatible with MongoDB 3.6.

When you use Amazon DocumentDB, you begin by creating a cluster. A cluster consists of zero or more database instances and a cluster volume that manages the data for those instances. An Amazon DocumentDB cluster volume is a virtual database storage volume that spans multiple Availability Zones. Each Availability Zone has a copy of the cluster data.

An Amazon DocumentDB cluster consists of two components:

- Cluster volume—Uses a cloud-native storage service to replicate data six ways across three Availability Zones, providing highly durable and available storage. An Amazon DocumentDB cluster has exactly one cluster volume, which can store up to 64 TB of data.
- Instances—Provide the processing power for the database, writing data to, and reading data from, the cluster storage volume. An Amazon DocumentDB cluster can have 0–16 instances.

Instances serve one of two roles:

- Primary instance—Supports read and write operations, and performs all the data modifications to the cluster volume. Each Amazon DocumentDB cluster has one primary instance.
- Replica instance—Supports only read operations. An Amazon DocumentDB cluster can have up to 15 replicas in addition to the primary instance. Having multiple replicas enables you to distribute read workloads. In addition, by placing replicas in separate Availability Zones, you also increase your cluster availability.

The following diagram illustrates the relationship between the cluster volume, the primary instance, and replicas in an Amazon DocumentDB cluster:

![image](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/docdb-endpoint-types.png?raw=true)


Cluster instances do not need to be of the same instance class, and they can be provisioned and terminated as desired. This architecture lets you scale your cluster’s compute capacity independently of its storage.

When your application writes data to the primary instance, the primary executes a durable write to the cluster volume. It then replicates the state of that write (not the data) to each active replica. Amazon DocumentDB replicas do not participate in processing writes, and thus Amazon DocumentDB replicas are advantageous for read scaling. Reads from Amazon DocumentDB replicas are eventually consistent with minimal replica lag—usually less than 100 milliseconds after the primary instance writes the data. Reads from the replicas are guaranteed to be read in the order in which they were written to the primary. Replica lag varies depending on the rate of data change, and periods of high write activity might increase the replica lag.

## Amazon DocumentDB Endpoints

Amazon DocumentDB provides multiple connection options to serve a wide range of use cases. To connect to an instance in an Amazon DocumentDB cluster, you specify the instance's endpoint. An endpoint is a host address and a port number, separated by a colon.

We recommend that you connect to your cluster using the cluster endpoint and in replica set mode (see Connecting to Amazon DocumentDB as a Replica Set) unless you have a specific use case for connecting to the reader endpoint or an instance endpoint. To route requests to your replicas, choose a driver read preference setting that maximizes read scaling while meeting your application's read consistency requirements. The secondaryPreferred read preference enables replica reads and frees up the primary instance to do more work.

The following endpoints are available from an Amazon DocumentDB cluster.

 ### Cluster Endpoint

The cluster endpoint connects to your cluster’s current primary instance. The cluster endpoint can be used for read and write operations. An Amazon DocumentDB cluster has exactly one cluster endpoint.

The cluster endpoint provides failover support for read and write connections to the cluster. If your cluster’s current primary instance fails, and your cluster has at least one active read replica, the cluster endpoint automatically redirects connection requests to a new primary instance. When connecting to your Amazon DocumentDB cluster, we recommend that you connect to your cluster using the cluster endpoint and in replica set mode. (https://docs.aws.amazon.com/documentdb/latest/developerguide/connect-to-replica-set.html)

https://docs.aws.amazon.com/documentdb/latest/developerguide/db-cluster-endpoints-find.html

### Reader Endpoint

Coming soon....

### Instance Endpoint

Something here but I don't know. Here is a Note:

#### Note
An instance’s role as primary or replica can change due to a failover event. Your applications should never assume that a particular instance endpoint is the primary instance. We do not recommend connecting to instance endpoints for production applications. Instead, we recommend that you connect to your cluster using the cluster endpoint and in replica set mode (see Connecting to Amazon DocumentDB as a Replica Set). For more advanced control of instance failover priority, see Understanding Amazon DocumentDB Cluster Fault Tolerance.

### Replica Set Mode

    Coming soon ...

### TLS Support
        
    Coming soon ...

### Amazon DocumentDB Storage

Amazon DocumentDB data is stored in a cluster volume, which is a single, virtual volume that uses solid state drives (SSDs). A cluster volume consists of six copies of your data, which are replicated automatically across multiple Availability Zones in a single AWS Region. This replication helps ensure that your data is highly durable, with less possibility of data loss. It also helps ensure that your cluster is more available during a failover because copies of your data already exist in other Availability Zones. These copies can continue to serve data requests to the instances in your Amazon DocumentDB cluster.

#### How Data Storage is Billed (Cost !!!!!!!!!!!!!)

Amazon DocumentDB automatically increases the size of a cluster volume as the amount of data increases. An Amazon DocumentDB cluster volume can grow to a maximum size of 64 TiB; however, you are only charged for the space that you use in an Amazon DocumentDB cluster volume. When Amazon DocumentDB data is removed, such as by dropping a table or partition, the overall allocated space remains the same. The free space is reused automatically when data volume increases in the future.

##### Note

Because storage costs are based on the storage "high water mark" (the maximum amount that was allocated for the Amazon DocumentDB cluster at any point in time), you can manage costs by avoiding ETL practices that create large volumes of temporary information, or that load large volumes of new data prior to removing unneeded older data.

### Amazon DocumentDB Replication

In an Amazon DocumentDB cluster, each replica instance exposes an independent endpoint. These replica endpoints provide read-only access to the data in the cluster volume. They enable you to scale the read workload for your data over multiple replicated instances. They also help improve the performance of data reads and increase the availability of the data in your Amazon DocumentDB cluster. Amazon DocumentDB replicas are also failover targets and are quickly promoted if the primary instance for your Amazon DocumentDB cluster fails.
    
### Amazon DocumentDB Reliability

Amazon DocumentDB is designed to be reliable, durable, and fault tolerant. (To improve availability, you should configure your Amazon DocumentDB cluster so that it has multiple replica instances in different Availability Zones.) Amazon DocumentDB includes several automatic features that make it a reliable database solution.

#### Storage Auto-Repair
#### Survivable Cache Warming
#### Crash Recovery
### Read Preference Options

Amazon DocumentDB uses a cloud-native shared storage service that replicates data six times across three Availability Zones to provide high levels of durability. Amazon DocumentDB does not rely on replicating data to multiple instances to achieve durability. Your cluster’s data is durable whether it contains a single instance or 15 instances.
    
#### Write Durability
#### Read Isolation
#### Read Consistency
#### Amazon DocumentDB Read Preferences

Amazon DocumentDB supports setting a read preference option only when reading data from the cluster endpoint in replica set mode. Setting a read preference option affects how your MongoDB client or driver routes read requests to instances in your Amazon DocumentDB cluster. You can set read preference options for a specific query, or as a general option in your MongoDB driver. (Consult your client or driver’s documentation for instructions on how to set a read preference option.)

If your client or driver is not connecting to an Amazon DocumentDB cluster endpoint in replica set mode, the result of specifying a read preference is undefined.

Amazon DocumentDB does not support setting tag sets as a read preference.

Supported Read Preference Options

- primary—Specifying a primary read preference helps ensure that all reads are routed to the cluster’s primary instance. If the primary instance is unavailable, the read operation fails. A primary read preference yields read-after-write consistency and is appropriate for use cases that prioritize read-after-write consistency over high availability and read scaling.

The following example specifies a primary read preference:

        ```
        db.example.find().readPref('primary')
        ```
 

- primaryPreferred—Specifying a primaryPreferred read preference routes reads to the primary instance under normal operation. If there is a primary failover, the client routes requests to a replica. A primaryPreferred read preference yields read-after-write consistency during normal operation, and eventually consistent reads during a failover event. A primaryPreferred read preference is appropriate for use cases that prioritize read-after-write consistency over read scaling, but still require high availability.

The following example specifies a primaryPreferred read preference:
```
db.example.find().readPref('primaryPreferred')
```

- secondary—Specifying a secondary read preference ensures that reads are only routed to a replica, never the primary instance. If there are no replica instances in a cluster, the read request fails. A secondary read preference yields eventually consistent reads and is appropriate for use cases that prioritize primary instance write throughput over high availability and read-after-write consistency.

The following example specifies a secondary read preference:
```
        db.example.find().readPref('secondary')
 ```

- secondaryPreferred—Specifying a secondaryPreferred read preference ensures that reads are routed to a read replica when one or more replicas are active. If there are no active replica instances in a cluster, the read request is routed to the primary instance. A secondaryPreferred read preference yields eventually consistent reads when the read is serviced by a read replica. It yields read-after-write consistency when the read is serviced by the primary instance (barring failover events). A secondaryPreferred read preference is appropriate for use cases that prioritize read scaling and high availability over read-after-write consistency.

The following example specifies a secondaryPreferred read preference:
```
        db.example.find().readPref('secondaryPreferred')
```

- nearest—Specifying a nearest read preference routes reads based solely on the measured latency between the client and all instances in the Amazon DocumentDB cluster. A nearest read preference yields eventually consistent reads when the read is serviced by a read replica. It yields read-after-write consistency when the read is serviced by the primary instance (barring failover events). A nearest read preference is appropriate for use cases that prioritize achieving the lowest possible read latency and high availability over read-after-write consistency and read scaling.

The following example specifies a nearest read preference:
```
        db.example.find().readPref('nearest')
```
#### High Availability
#### Scaling Reads

### TTL Deletes

## 3. What is a Document Database?

## Document Database Use Cases

#### User Profiles
Because document databases have a flexible schema, they can store documents that have different attributes and data values. Document databases are a practical solution to online profiles in which different users provide different types of information. Using a document database, you can store each user's profile efficiently by storing only the attributes that are specific to each user.

Suppose that a user elects to add or remove information from their profile. In this case, their document could be easily replaced with an updated version that contains any recently added attributes and data or omits any newly omitted attributes and data. Document databases easily manage this level of individuality and fluidity.

### Real-Time Big Data

Historically, the ability to extract information from operational data was hampered by the fact that operational databases and analytical databases were maintained in different environments—operational and business/reporting respectively. Being able to extract operational information in real time is critical in a highly competitive business environment. By using document databases, a business can store and manage operational data from any source and concurrently feed the data to the BI engine of choice for analysis. 

There is no requirement to have two environments.

### Content Management
To effectively manage content, you must be able to collect and aggregate content from a variety of sources, and then deliver it to the customer. Due to their flexible schema, document databases are perfect for collecting and storing any type of data. You can use them to create and incorporate new types of content, including user-generated content, such as images, comments, and videos.

## Understanding Documents 
    Here :))))))))))) 
        https://docs.aws.amazon.com/documentdb/latest/developerguide/document-database-documents-understanding.html

## Working with Documents 
    https://docs.aws.amazon.com/documentdb/latest/developerguide/document-database-working-with-documents.html

## 4. Amazon DocumentDB Quick Start Using AWS CloudFormation

(something stupid :))


    


