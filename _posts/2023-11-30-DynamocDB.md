---
title: Amazon DynamoDB
# author:
#   name: Life Zero
#   link: https://github.com/lacie-life
date:  2023-11-30 11:11:14 +0700
categories: [Devops]
tags: [Tutorial]
img_path: /assets/img/post_assest/pvo/
render_with_liquid: false
---

# DynamoDB is something Magic

## 1. What Is Amazon DynamoDB?

Amazon DynamoDB is a fully managed NoSQL database service that provides fast and predictable performance with seamless scalability. DynamoDB lets you offload the administrative burdens of operating and scaling a distributed database so that you don't have to worry about hardware provisioning, setup and configuration, replication, software patching, or cluster scaling.

## 2. Core Components of Amazon DynamoDB

In DynamoDB, tables, items, and attributes are the core components that you work with. A table is a collection of items, and each item is a collection of attributes. DynamoDB uses primary keys to uniquely identify each item in a table and secondary indexes to provide more querying flexibility. You can use DynamoDB Streams to capture data modification events in DynamoDB tables.

## a. Tables, Items, and Attributes

- Tables – Similar to other database systems, DynamoDB stores data in tables. A table is a collection of data.
- Items – Each table contains zero or more items. An item is a group of attributes that is uniquely identifiable among all of the other items. 
- Attributes – Each item is composed of one or more attributes. An attribute is a fundamental data element, something that does not need to be broken down any further

![Example Tables, Items, and Attributes](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/People.png?raw=true)


## b. Primary Key

When you create a table, in addition to the table name, you must specify the primary key of the table. The primary key uniquely identifies each item in the table, so that no two items can have the same key.

DynamoDB supports two different kinds of primary keys:

- Partition key – A simple primary key, composed of one attribute known as the partition key. 

DynamoDB uses the partition key's value as input to an internal hash function. The output from the hash function determines the partition (physical storage internal to DynamoDB) in which the item will be stored.

In a table that has only a partition key, no two items can have the same partition key value.

- Partition key and sort key – Referred to as a composite primary key, this type of key is composed of two attributes. The first attribute is the partition key, and the second attribute is the sort key.

DynamoDB uses the partition key value as input to an internal hash function. The output from the hash function determines the partition (physical storage internal to DynamoDB) in which the item will be stored. All items with the same partition key value are stored together, in sorted order by sort key value.

In a table that has a partition key and a sort key, it's possible for two items to have the same partition key value. However, those two items must have different sort key values.

### Note
The partition key of an item is also known as its hash attribute. The term hash attribute derives from the use of an internal hash function in DynamoDB that evenly distributes data items across partitions, based on their partition key values.

The sort key of an item is also known as its range attribute. The term range attribute derives from the way DynamoDB stores items with the same partition key physically close together, in sorted order by the sort key value.

## c. Secondary Indexes

You can create one or more secondary indexes on a table. A secondary index lets you query the data in the table using an alternate key, in addition to queries against the primary key. DynamoDB doesn't require that you use indexes, but they give your applications more flexibility when querying your data. After you create a secondary index on a table, you can read data from the index in much the same way as you do from the table.

DynamoDB supports two kinds of indexes:

- Global secondary index – An index with a partition key and sort key that can be different from those on the table.

- Local secondary index – An index that has the same partition key as the table, but a different sort key.

Each table in DynamoDB has a quota of 20 global secondary indexes (default quota) and 5 local secondary indexes per table.

![Secondary Indexes](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/GenreAlbumTitle.png?raw=true)

## d. DynamoDB Streams

DynamoDB Streams is an optional feature that captures data modification events in DynamoDB tables. The data about these events appear in the stream in near-real time, and in the order that the events occurred.

Each event is represented by a stream record. If you enable a stream on a table, DynamoDB Streams writes a stream record whenever one of the following events occurs:

- A new item is added to the table: The stream captures an image of the entire item, including all of its attributes.
- An item is updated: The stream captures the "before" and "after" image of any attributes that were modified in the item.
- An item is deleted from the table: The stream captures an image of the entire item before it was deleted.

Each stream record also contains the name of the table, the event timestamp, and other metadata. Stream records have a lifetime of 24 hours; after that, they are automatically removed from the stream.

You can use DynamoDB Streams together with AWS Lambda to create a trigger—code that executes automatically whenever an event of interest appears in a stream. For example, consider a Customers table that contains customer information for a company. Suppose that you want to send a "welcome" email to each new customer. You could enable a stream on that table, and then associate the stream with a Lambda function. The Lambda function would execute whenever a new stream record appears, but only process new items added to the Customers table. For any item that has an EmailAddress attribute, the Lambda function would invoke Amazon Simple Email Service (Amazon SES) to send an email to that address.

![Stream](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/Streams.png?raw=true)

### Note
In this example, the last customer, Craig Roe, will not receive an email because he doesn't have an EmailAddress.

In addition to triggers, DynamoDB Streams enables powerful solutions such as data replication within and across AWS Regions, materialized views of data in DynamoDB tables, data analysis using Kinesis materialized views, and much more.

## 3. DynamoDB API

## a. Control Plane

Control plane operations let you create and manage DynamoDB tables. They also let you work with indexes, streams, and other objects that are dependent on tables.

- CreateTable – Creates a new table. Optionally, you can create one or more secondary indexes, and enable DynamoDB Streams for the table.
- DescribeTable– Returns information about a table, such as its primary key schema, throughput settings, and index information.
- ListTables – Returns the names of all of your tables in a list.
- UpdateTable – Modifies the settings of a table or its indexes, creates or removes new indexes on a table, or modifies DynamoDB Streams settings for a table.
- DeleteTable – Removes a table and all of its dependent objects from DynamoDB.

## b. Data Plane

Data plane operations let you perform create, read, update, and delete (also called CRUD) actions on data in a table. Some of the data plane operations also let you read data from a secondary index.

### Creating Data

- PutItem – Writes a single item to a table. You must specify the primary key attributes, but you don't have to specify other attributes.
- BatchWriteItem – Writes up to 25 items to a table. This is more efficient than calling PutItem multiple times because your application only needs a single network round trip to write the items. You can also use BatchWriteItem for deleting multiple items from one or more tables.

### Reading Data

- GetItem – Retrieves a single item from a table. You must specify the primary key for the item that you want. You can retrieve the entire item, or just a subset of its attributes.
- BatchGetItem – Retrieves up to 100 items from one or more tables. This is more efficient than calling GetItem multiple times because your application only needs a single network round trip to read the items.
- Query – Retrieves all items that have a specific partition key. You must specify the partition key value. You can retrieve entire items, or just a subset of their attributes. Optionally, you can apply a condition to the sort key values so that you only retrieve a subset of the data that has the same partition key. You can use this operation on a table, provided that the table has both a partition key and a sort key. You can also use this operation on an index, provided that the index has both a partition key and a sort key.
- Scan – Retrieves all items in the specified table or index. You can retrieve entire items, or just a subset of their attributes. Optionally, you can apply a filtering condition to return only the values that you are interested in and discard the rest.

### Updating Data

- UpdateItem – Modifies one or more attributes in an item. You must specify the primary key for the item that you want to modify. You can add new attributes and modify or remove existing attributes. You can also perform conditional updates, so that the update is only successful when a user-defined condition is met. Optionally, you can implement an atomic counter, which increments or decrements a numeric attribute without interfering with other write requests.

### Deleting Data

- DeleteItem – Deletes a single item from a table. You must specify the primary key for the item that you want to delete.
- BatchWriteItem – Deletes up to 25 items from one or more tables. This is more efficient than calling DeleteItem multiple times because your application only needs a single network round trip to delete the items. You can also use BatchWriteItem for adding multiple items to one or more tables.

## c. DynamoDB Streams

DynamoDB Streams operations let you enable or disable a stream on a table, and allow access to the data modification records contained in a stream.

- ListStreams – Returns a list of all your streams, or just the stream for a specific table.
- DescribeStream – Returns information about a stream, such as its Amazon Resource Name (ARN) and where your application can begin reading the first few stream records.
- GetShardIterator – Returns a shard iterator, which is a data structure that your application uses to retrieve the records from the stream.
- GetRecords – Retrieves one or more stream records, using a given shard iterator.

## d. Transactions

Transactions provide atomicity, consistency, isolation, and durability (ACID) enabling you to maintain data correctness in your applications more easily.

- TransactWriteItems – A batch operation that allows Put, Update, and Delete operations to multiple items both within and across tables with a guaranteed all-or-nothing result.
- TransactGetItems – A batch operation that allows Get operations to retrieve multiple items from one or more tables.

## 4. Naming Rules and Data Types

## a. Naming Rules

Tables, attributes, and other objects in DynamoDB must have names. Names should be meaningful and concise—for example, names such as Products, Books, and Authors are self-explanatory.

The following are the naming rules for DynamoDB:

All names must be encoded using UTF-8, and are case-sensitive.

Table names and index names must be between 3 and 255 characters long, and can contain only the following characters:

- a-z
- A-Z
- 0-9
- _ (underscore)
- \- (dash)
- . (dot)

Attribute names must be between 1 and 255 characters long.

## b. Data Types

DynamoDB supports many different data types for attributes within a table. They can be categorized as follows:

- Scalar Types – A scalar type can represent exactly one value. The scalar types are number, string, binary, Boolean, and null.

- Document Types – A document type can represent a complex structure with nested attributes, such as you would find in a JSON document. The document types are list and map.

- Set Types – A set type can represent multiple scalar values. The set types are string set, number set, and binary set.

When you create a table or a secondary index, you must specify the names and data types of each primary key attribute (partition key and sort key). Furthermore, each primary key attribute must be defined as type string, number, or binary.

DynamoDB is a NoSQL database and is schemaless. This means that, other than the primary key attributes, you don't have to define any attributes or data types when you create tables. By comparison, relational databases require you to define the names and data types of each column when you create a table.

The following are descriptions of each data type, along with examples in JSON format.

(More details: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.NamingRulesDataTypes.html)

## 5. Read Consistency

Amazon DynamoDB is available in multiple AWS Regions around the world. Each Region is independent and isolated from other AWS Regions. For example, if you have a table called People in the us-east-2 Region and another table named People in the us-west-2 Region, these are considered two entirely separate tables. For a list of all the AWS Regions in which DynamoDB is available, see AWS Regions and Endpoints in the Amazon Web Services General Reference.

Every AWS Region consists of multiple distinct locations called Availability Zones. Each Availability Zone is isolated from failures in other Availability Zones, and provides inexpensive, low-latency network connectivity to other Availability Zones in the same Region. This allows rapid replication of your data among multiple Availability Zones in a Region.

When your application writes data to a DynamoDB table and receives an HTTP 200 response (OK), the write has occurred and is durable. The data is eventually consistent across all storage locations, usually within one second or less.

DynamoDB supports eventually consistent and strongly consistent reads.

### Eventually Consistent Reads

When you read data from a DynamoDB table, the response might not reflect the results of a recently completed write operation. The response might include some stale data. If you repeat your read request after a short time, the response should return the latest data.

### Strongly Consistent Reads

When you request a strongly consistent read, DynamoDB returns a response with the most up-to-date data, reflecting the updates from all prior write operations that were successful. However, this consistency comes with some disadvantages:

- A strongly consistent read might not be available if there is a network delay or outage. In this case, DynamoDB may return a server error (HTTP 500).

- Strongly consistent reads may have higher latency than eventually consistent reads.

- Strongly consistent reads are not supported on global secondary indexes.

- Strongly consistent reads use more throughput capacity than eventually consistent reads. More details: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.ReadWriteCapacityMode.html

### Note
DynamoDB uses eventually consistent reads, unless you specify otherwise. Read operations (such as GetItem, Query, and Scan) provide a ConsistentRead parameter. If you set this parameter to true, DynamoDB uses strongly consistent reads during the operation.

## 6. Partitions and Data Distribution

Amazon DynamoDB stores data in partitions. A partition is an allocation of storage for a table, backed by solid state drives (SSDs) and automatically replicated across multiple Availability Zones within an AWS Region. Partition management is handled entirely by DynamoDB—you never have to manage partitions yourself.

When you create a table, the initial status of the table is CREATING. During this phase, DynamoDB allocates sufficient partitions to the table so that it can handle your provisioned throughput requirements. You can begin writing and reading table data after the table status changes to ACTIVE.

DynamoDB allocates additional partitions to a table in the following situations:

- If you increase the table's provisioned throughput settings beyond what the existing partitions can support.
- If an existing partition fills to capacity and more storage space is required.

Partition management occurs automatically in the background and is transparent to your applications. Your table remains available throughout and fully supports your provisioned throughput requirements.

Global secondary indexes in DynamoDB are also composed of partitions. The data in a global secondary index is stored separately from the data in its base table, but index partitions behave in much the same way as table partitions.

### Data Distribution: Partition Key

If your table has a simple primary key (partition key only), DynamoDB stores and retrieves each item based on its partition key value.

To write an item to the table, DynamoDB uses the value of the partition key as input to an internal hash function. The output value from the hash function determines the partition in which the item will be stored.

To read an item from the table, you must specify the partition key value for the item. DynamoDB uses this value as input to its hash function, yielding the partition in which the item can be found.

The following diagram shows a table named Pets, which spans multiple partitions. The table's primary key is AnimalType (only this key attribute is shown). DynamoDB uses its hash function to determine where to store a new item, in this case based on the hash value of the string Dog. Note that the items are not stored in sorted order. Each item's location is determined by the hash value of its partition key.

![Partition Key](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/PartitionKey.png?raw=true)

#### Note
DynamoDB is optimized for uniform distribution of items across a table's partitions, no matter how many partitions there may be. We recommend that you choose a partition key that can have a large number of distinct values relative to the number of items in the table.

### Data Distribution: Partition Key and Sort Key

If the table has a composite primary key (partition key and sort key), DynamoDB calculates the hash value of the partition key in the same way as described in Data Distribution: Partition Key. However, it stores all the items with the same partition key value physically close together, ordered by sort key value.

To write an item to the table, DynamoDB calculates the hash value of the partition key to determine which partition should contain the item. In that partition, several items could have the same partition key value. So DynamoDB stores the item among the others with the same partition key, in ascending order by sort key.

To read an item from the table, you must specify its partition key value and sort key value. DynamoDB calculates the partition key's hash value, yielding the partition in which the item can be found.

You can read multiple items from the table in a single operation (Query) if the items you want have the same partition key value. DynamoDB returns all of the items with that partition key value. Optionally, you can apply a condition to the sort key so that it returns only the items within a certain range of values.

Suppose that the Pets table has a composite primary key consisting of AnimalType (partition key) and Name (sort key). The following diagram shows DynamoDB writing an item with a partition key value of Dog and a sort key value of Fido.

![Partition Key and Sort Key](https://github.com/lacie-life/lacie-life.github.io/blob/main/assets/img/post_assest/PartitionKeySortKey.png?raw=true)

To read that same item from the Pets table, DynamoDB calculates the hash value of Dog, yielding the partition in which these items are stored. DynamoDB then scans the sort key attribute values until it finds Fido.

To read all of the items with an AnimalType of Dog, you can issue a Query operation without specifying a sort key condition. By default, the items are returned in the order that they are stored (that is, in ascending order by sort key). Optionally, you can request descending order instead.

To query only some of the Dog items, you can apply a condition to the sort key (for example, only the Dog items where Name begins with a letter that is within the range A through K).

#### Note
In a DynamoDB table, there is no upper limit on the number of distinct sort key values per partition key value. If you needed to store many billions of Dog items in the Pets table, DynamoDB would allocate enough storage to handle this requirement automatically.

## 7. Setting Up DynamoDB (Web Service)

## Signing Up for AWS

To use the DynamoDB service, you must have an AWS account. If you don't already have an account, you are prompted to create one when you sign up. You're not charged for any AWS services that you sign up for unless you use them.

#### To sign up for AWS

- Step 1: Open https://portal.aws.amazon.com/billing/signup
- Step 2: Follow the online instructions.
Part of the sign-up procedure involves receiving a phone call and entering a verification code on the phone keypad.

#### Getting an AWS Access Key

Before you can access DynamoDB programmatically or through the AWS Command Line Interface (AWS CLI), you must have an AWS access key. You don't need an access key if you plan to use the DynamoDB console only.

Access keys consist of an access key ID and secret access key, which are used to sign programmatic requests that you make to AWS. If you don't have access keys, you can create them from the AWS Management Console. As a best practice, do not use the AWS account root user access keys for any task where it's not required. Instead, create a new administrator IAM user with access keys for yourself.

The only time that you can view or download the secret access key is when you create the keys. You cannot recover them later. However, you can create new access keys at any time. You must also have permissions to perform the required IAM actions. For more information, see Permissions Required to Access IAM Resources in the IAM User Guide.

##### To create access keys for an IAM user

- Sign in to the AWS Management Console and open the IAM console at https://console.aws.amazon.com/iam/.

- In the navigation pane, choose Users.

- Choose the name of the user whose access keys you want to create, and then choose the Security credentials tab.

- In the Access keys section, choose Create access key.

- To view the new access key pair, choose Show. You will not have access to the secret access key again after this dialog box closes.

- To download the key pair, choose Download .csv file. Store the keys in a secure location. You will not have access to the secret access key again after this dialog box closes.

- After you download the .csv file, choose Close. When you create an access key, the key pair is active by default, and you can use the pair right away.

   
#### Configuring Your Credentials

Before you can access DynamoDB programmatically or through the AWS CLI, you must configure your credentials to enable authorization for your applications.

There are several ways to do this. For example, you can manually create the credentials file to store your access key ID and secret access key. You also can use the aws configure command of the AWS CLI to automatically create the file. Alternatively, you can use environment variables. For more information about configuring your credentials, see the programming-specific AWS SDK developer guide.

To install and configure the AWS CLI, see Using the AWS CLI.

## 8.  Accessing DynamoDB

## Using the Console
You can access the AWS Management Console for Amazon DynamoDB at https://console.aws.amazon.com/dynamodb/home.

You can use the console to do the following in DynamoDB:

- Monitor recent alerts, total capacity, service health, and the latest DynamoDB news on the DynamoDB dashboard.
- Create, update, and delete tables. The capacity calculator provides estimates of how many capacity units to request based on the usage information you provide.
- Manage streams.
- View, add, update, and delete items that are stored in tables. Manage Time to Live (TTL) to define when items in a table expire so that they can be automatically deleted from the database.
- Query and scan a table.
- Set up and view alarms to monitor your table's capacity usage. View your table's top monitoring metrics on real-time graphs from CloudWatch.
- Modify a table's provisioned capacity.
- Create and delete global secondary indexes.
- Create triggers to connect DynamoDB streams to AWS Lambda functions.
- Apply tags to your resources to help organize and identify them.
- Purchase reserved capacity.
The console displays an introductory screen that prompts you to create your first table. To view your tables, in the navigation pane on the left side of the console, choose Tables.
Here's a high-level overview of the actions available per table within each navigation tab:
- Overview – View stream and table details, and manage streams and Time to Live (TTL).
- Items – Manage items and perform queries and scans.
- Metrics – Monitor Amazon CloudWatch metrics.
- Alarms – Manage CloudWatch alarms.
- Capacity – Modify a table's provisioned capacity.
- Indexes – Manage global secondary indexes.
- Triggers – Manage triggers to connect DynamoDB streams to Lambda functions.
- Access control – Set up fine-grained access control with web identity federation.
- Tags – Apply tags to your resources to help organize and identify them.

## Connecting to your Linux instance from Windows using PuTTY: 
    https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html?icmpid=docs_ec2_console

## Installing the AWS CLI version 2 on Linux
    https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html

## Python API

    https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/GettingStarted.Python.html

