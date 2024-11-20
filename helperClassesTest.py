import torch
import numpy as np
from scipy.spatial import distance
import re
import sys, os

class Utils:
    def get_schema_table_table_definition_map(file_path):
      with open(file_path, 'r') as file:
          lines = file.readlines()

      current_schema = None
      current_table = None
      table_definition = []
      schema_tables = {}

      for line in lines:
          line = line.strip()

          if line.startswith('CREATE SCHEMA'):
              if current_schema and current_table:
                  if current_schema not in schema_tables:
                      schema_tables[current_schema] = {}
                  # Join with a space instead of a newline
                  schema_tables[current_schema][current_table] = ' '.join(table_definition)

              current_schema = line.split()[2]
              current_table = None
              table_definition = []

          elif line.startswith('CREATE TABLE'):
              if current_table:
                  if current_schema not in schema_tables:
                      schema_tables[current_schema] = {}
                  # Join with a space instead of a newline
                  schema_tables[current_schema][current_table] = ' '.join(table_definition)

              current_table = line.split()[2].split("(")[0]
              table_definition = [line]

          elif current_table:
              table_definition.append(line)

          if line.endswith(';') and current_table:
              if current_schema not in schema_tables:
                  schema_tables[current_schema] = {}
              # Join with a space instead of a newline
              schema_tables[current_schema][current_table] = ' '.join(table_definition)
              current_table = None
              table_definition = []

      return schema_tables

    @staticmethod
    def split_camel_case(input_string):
      """
      Splits a camel case string into separate words.
    
      Parameters:
      input_string (str): The camel case string to be split.

      Returns:
      list: A list of words split from the camel case string.
      """
      words = []
      current_word = ""

      for char in input_string:
          # Check if the character is uppercase and the current word is not empty
          if char.isupper() and current_word:
              # Append the current word to words and start a new word
              words.append(current_word)
              current_word = char
          else:
              # Append the character to the current word
              current_word += char

      # Append the last word if it's not empty
      if current_word:
          words.append(current_word)

      return words

    @staticmethod
    def get_embeddings(sequence, model, tokenizer, max_len = None):
      """
      Returns embeddings as numpy arrays, using the model and tokenizer provided
      """
      input_ids = None
      if max_len == None:
        input_ids = tokenizer.encode(sequence, return_tensors='pt')
      else:
        input_ids = tokenizer.encode(sequence, return_tensors='pt', padding=True, truncation=True, max_length=max_len, add_special_tokens = True)
      print(len(input_ids[0]))
      with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state
      return embeddings[0][0:-1].cpu().numpy()

    @staticmethod
    def calculate_embeddings_for_words(word_list, model, tokenizer, split_camel_case_words):
      words_to_embeddings = {}
      for word in word_list:
        current_word = word
        current_embeddings = []
        if split_camel_case_words:
          for word1 in Utils.split_camel_case(word):
            embeddings = Utils.get_embeddings(word1, model, tokenizer)
            current_embeddings.append(embeddings[1])
        else:
          embeddings = Utils.get_embeddings(word, model, tokenizer)
          current_embeddings.append(embeddings[1])
        words_to_embeddings[current_word] = current_embeddings
      
      return words_to_embeddings

    @staticmethod
    def calculate_min_euclidean_distance_of_word_embeddings(question_embeddings_dict, table_embeddings_dict):
      """
      Calculates the minimum Euclidean distance of any embedding for each table 
      (for example, table "PurchaseOrder" will have two embeddings, one for 
      "Purchase" and one for "Order") with any embedding from the question. 
      It returns a dictionary, with table names as keys and the minimum distance 
      found as values.
      """

      table_min_dist = {}

      for table, t_embeddings in table_embeddings_dict.items():
        min_distance = np.inf
        for t_embedding in t_embeddings:
          for q_word, q_embeddings in question_embeddings_dict.items():
            for q_embedding in q_embeddings:
              dist = distance.euclidean(t_embedding, q_embedding)
              if dist < min_distance:
                min_distance = dist
        table_min_dist[table] = min_distance

      return table_min_dist
  
    @staticmethod
    def calculate_max_cosine_similarity_of_word_embeddings(question_embeddings_dict, table_embeddings_dict):
      """
      Calculates the minimum Euclidean distance of any embedding for each table 
      (for example, table "PurchaseOrder" will have two embeddings, one for 
      "Purchase" and one for "Order") with any embedding from the question. 
      It returns a dictionary, with table names as keys and the minimum distance 
      found as values.
      """

      table_max_sim = {}

      for table, t_embeddings in table_embeddings_dict.items():
        max_similarity = -np.inf
        for t_embedding in t_embeddings:
          for q_word, q_embeddings in question_embeddings_dict.items():
            for q_embedding in q_embeddings:
              similarity = 1 - distance.cosine(t_embedding, q_embedding)
              if similarity > max_similarity:
                max_similarity = similarity
        table_max_sim[table] = max_similarity

      return table_max_sim

    @staticmethod
    def calculate_cosine_similarity(vector_1, vector_2):
      """
      Calculates cosine similarity of 2 vectors 
      found as values.
      """
      return 1 - distance.cosine(vector_1, vector_2)

    @staticmethod
    def get_community_avg_metric(coms, table_minimum_distances, remove_schema_name_from_com_members):
      """
      Calculates the average distance of a community, given the distances of 
      individual members of the community. 
      """
      community_avg_dist = {}
      for community in coms.communities:
        community_table_distances = []
        for table_name in community:
          final_table_name = table_name
          if remove_schema_name_from_com_members:
            final_table_name = table_name.split('.')[1]
          community_table_distances.append(table_minimum_distances[final_table_name])
        community_avg_dist[tuple(community)] = np.mean(community_table_distances)
      return community_avg_dist
  
    @staticmethod
    def get_adventureworks_2014_mod_schema():
        """
        Returns full schema of adventureworks 2014 modified, using classes Database, Schema, Table, TableRelation and TableField

        """
        database = Database(
                            'AdventureWorks2014_mod', 
                            [
                                Schema(
                                        'Person', 
                                        [
                                            Table(
                                                'Person', 
                                                'BusinessEntity', 
                                                [
                                                    TableField('BusinessEntityID', 'SERIAL', ['PRIMARY KEY'], 'Primary key for all customers, vendors, and employees.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_BusinessEntity_rowguid" DEFAULT (uuid_generate_v1())'], None),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_BusinessEntity_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Source of the ID that connects vendors, customers, and employees with address and contact information.'
                                            ),
                                            Table(
                                                'Person', 
                                                'Person', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for Person records.'), 
                                                    TableField('PersonType', 'char(2)', ['NOT NULL'], 'Primary type of person: SC = Store Contact, IN = Individual (retail) customer, SP = Sales person, EM = Employee (non-sales), VC = Vendor contact, GC = General contact'), 
                                                    TableField('NameStyle', 'boolean', ['NOT NULL', 'CONSTRAINT "DF_Person_NameStyle" DEFAULT (false)'], '0 = The data in FirstName and LastName are stored in western style (first name, last name) order.  1 = Eastern style (last name, first name) order.'), 
                                                    TableField('Title', 'varchar(8)', ['NULL'], 'A courtesy title. For example, Mr. or Ms.'), 
                                                    TableField('FirstName', 'varchar(50)', ['NOT NULL'], 'First name of the person.'), 
                                                    TableField('MiddleName', 'varchar(50)', ['NULL'], 'Middle name or middle initial of the person.'), 
                                                    TableField('LastName', 'varchar(50)', ['NOT NULL'], 'Last name of the person.'), 
                                                    TableField('Suffix', 'varchar(10)', ['NULL'], 'Surname suffix. For example, Sr. or Jr.'), 
                                                    TableField('EmailPromotion', 'INT', ['NOT NULL', 'CONSTRAINT "DF_Person_EmailPromotion" DEFAULT (0)'], '0 = Contact does not wish to receive e-mail promotions, 1 = Contact does wish to receive e-mail promotions from AdventureWorks, 2 = Contact does wish to receive e-mail promotions from AdventureWorks and selected partners.'), 
                                                    TableField('AdditionalContactInfo', 'XML', ['NULL'], 'Additional contact information about the person stored in xml format.'), 
                                                    TableField('Demographics', 'XML', ['NULL'], 'Personal information such as hobbies, and income collected from online shoppers. Used for sales analysis.'), 
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_Person_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Person_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'BusinessEntity', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES Person.BusinessEntity(BusinessEntityID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_Person_EmailPromotion" CHECK (EmailPromotion BETWEEN 0 AND 2)',
                                                    'CONSTRAINT "CK_Person_PersonType" CHECK (PersonType IS NULL OR UPPER(PersonType) IN (\'SC\', \'VC\', \'IN\', \'EM\', \'SP\', \'GC\'))',
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES Person.BusinessEntity(BusinessEntityID)'
                                                ],
                                                'Human beings involved with AdventureWorks: employees, customer contacts, and vendor contacts.'
                                            ),
                                            Table(
                                                'Person', 
                                                'StateProvince', 
                                                [
                                                    TableField('StateProvinceID', 'SERIAL', ['PRIMARY KEY'], 'Primary key for StateProvince records.'), 
                                                    TableField('StateProvinceCode', 'char(3)', ['NOT NULL'], 'ISO standard state or province code.'), 
                                                    TableField('CountryRegionCode', 'varchar(3)', ['NOT NULL'], 'ISO standard country or region code. Foreign key to CountryRegion.CountryRegionCode.'), 
                                                    TableField('IsOnlyStateProvinceFlag', 'boolean', ['NOT NULL', 'CONSTRAINT "DF_StateProvince_IsOnlyStateProvinceFlag" DEFAULT (true)'], '0 = StateProvinceCode exists. 1 = StateProvinceCode unavailable, using CountryRegionCode.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'State or province description.'), 
                                                    TableField('TerritoryID', 'INT', ['NOT NULL'], 'ID of the territory in which the state or province is located. Foreign key to SalesTerritory.SalesTerritoryID.'), 
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_StateProvince_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_StateProvince_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'CountryRegion', 'CountryRegionCode', 'CountryRegionCode', 'FOREIGN KEY (CountryRegionCode) REFERENCES Person.CountryRegion(CountryRegionCode)'),
                                                    TableRelation('Sales', 'SalesTerritory', 'TerritoryID', 'TerritoryID', 'FOREIGN KEY (TerritoryID) REFERENCES Sales.SalesTerritory(TerritoryID)')
                                                ],
                                                [
                                                    'FOREIGN KEY (CountryRegionCode) REFERENCES Person.CountryRegion(CountryRegionCode)',
                                                    'FOREIGN KEY (TerritoryID) REFERENCES Sales.SalesTerritory(TerritoryID)'
                                                ],
                                                'State and province lookup table.'
                                            ),
                                            Table(
                                                'Person', 
                                                'Address', 
                                                [
                                                    TableField('AddressID', 'SERIAL', ['PRIMARY KEY'], 'Primary key for Address records.'), 
                                                    TableField('AddressLine1', 'varchar(60)', ['NOT NULL'], 'First street address line.'), 
                                                    TableField('AddressLine2', 'varchar(60)', ['NULL'], 'Second street address line.'), 
                                                    TableField('City', 'varchar(30)', ['NOT NULL', 'CONSTRAINT "DF_StateProvince_IsOnlyStateProvinceFlag" DEFAULT (true)'], 'Name of the city.'), 
                                                    TableField('StateProvinceID', 'INT', ['NOT NULL'], 'Unique identification number for the state or province. Foreign key to StateProvince table.'), 
                                                    TableField('PostalCode', 'varchar(15)', ['NOT NULL'], 'Postal code for the street address.'), 
                                                    TableField('SpatialLocation', 'bytea', ['NULL'], 'Latitude and longitude of this address.'), 
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_Address_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Address_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'StateProvince', 'StateProvinceID', 'StateProvinceID', 'FOREIGN KEY (StateProvinceID) REFERENCES Person.StateProvince(StateProvinceID)')
                                                ],
                                                [
                                                    'FOREIGN KEY (StateProvinceID) REFERENCES Person.StateProvince(StateProvinceID)'
                                                ],
                                                'Street address information for customers, employees, and vendors.'
                                            ),
                                            Table(
                                                'Person', 
                                                'AddressType', 
                                                [
                                                    TableField('AddressTypeID', 'SERIAL', ['PRIMARY KEY'], 'Primary key for AddressType records.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Address type description. For example, Billing, Home, or Shipping.'), 
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_AddressType_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_AddressType_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Types of addresses stored in the Address table.'
                                            ),
                                            Table(
                                                'Person', 
                                                'BusinessEntityAddress', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to BusinessEntity.BusinessEntityID.'), 
                                                    TableField('AddressID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to Address.AddressID.'), 
                                                    TableField('AddressTypeID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to AddressType.AddressTypeID.'), 
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_BusinessEntityAddress_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_BusinessEntityAddress_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'Address', 'AddressID', 'AddressID', 'FOREIGN KEY (AddressID) REFERENCES Person.Address(AddressID)'),
                                                    TableRelation('Person', 'AddressType', 'AddressTypeID', 'AddressTypeID', 'FOREIGN KEY (AddressTypeID) REFERENCES Person.AddressType(AddressTypeID)'),
                                                    TableRelation('Person', 'BusinessEntity', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES Person.BusinessEntity(BusinessEntityID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (BusinessEntityID, AddressID, AddressTypeID)',
                                                    'FOREIGN KEY (AddressID) REFERENCES Person.Address(AddressID)',
                                                    'FOREIGN KEY (AddressTypeID) REFERENCES Person.AddressType(AddressTypeID)',
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES Person.BusinessEntity(BusinessEntityID)'
                                                ],
                                                'Cross-reference table mapping customers, vendors, and employees to their addresses.'
                                            ),
                                            Table(
                                                'Person', 
                                                'ContactType', 
                                                [
                                                    TableField('ContactTypeID', 'SERIAL', ['PRIMARY KEY'], 'Primary key for ContactType records.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Contact type description.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ContactType_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Lookup table containing the types of business entity contacts.'
                                            ),
                                            Table(
                                                'Person', 
                                                'BusinessEntityContact', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to BusinessEntity.BusinessEntityID.'), 
                                                    TableField('PersonID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to Person.BusinessEntityID.'), 
                                                    TableField('ContactTypeID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to ContactType.ContactTypeID.'), 
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_BusinessEntityContact_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_BusinessEntityContact_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'Person', 'PersonID', 'BusinessEntityID', 'FOREIGN KEY (PersonID) REFERENCES Person.Person(BusinessEntityID)'),
                                                    TableRelation('Person', 'ContactType', 'ContactTypeID', 'ContactTypeID', 'FOREIGN KEY (ContactTypeID) REFERENCES Person.ContactType(ContactTypeID)'),
                                                    TableRelation('Person', 'BusinessEntity', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES Person.BusinessEntity(BusinessEntityID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (BusinessEntityID, PersonID, ContactTypeID)',
                                                    'FOREIGN KEY (PersonID) REFERENCES Person.Person(BusinessEntityID)',
                                                    'FOREIGN KEY (ContactTypeID) REFERENCES Person.ContactType(ContactTypeID)',
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES Person.BusinessEntity(BusinessEntityID)'
                                                ],
                                                'Cross-reference table mapping stores, vendors, and employees to people'
                                            ),
                                            Table(
                                                'Person', 
                                                'EmailAddress', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL'], 'Primary key. Person associated with this email address. Foreign key to Person.BusinessEntityID'), 
                                                    TableField('EmailAddressID', 'SERIAL', None, 'Primary key. ID of this email address.'), 
                                                    TableField('EmailAddress', 'varchar(50)', ['NULL'], 'E-mail address for the person.'), 
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_EmailAddress_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_EmailAddress_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'Person', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES Person.Person(BusinessEntityID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (BusinessEntityID, EmailAddressID)',
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES Person.Person(BusinessEntityID)'
                                                ],
                                                'Where to send a person email.'
                                            ),
                                            Table(
                                                'Person', 
                                                'Password', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL', 'PRIMARY KEY'], 'Primary key. Person associated with this password. Foreign key to Person.BusinessEntityID'), 
                                                    TableField('PasswordHash', 'VARCHAR(128)', ['NOT NULL'], 'Password for the e-mail account.'), 
                                                    TableField('PasswordSalt', 'VARCHAR(10)', ['NOT NULL'], 'Random value concatenated with the password string before the password is hashed.'), 
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_Password_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Password_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'Person', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES Person.Person(BusinessEntityID)')
                                                ],
                                                [
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES Person.Person(BusinessEntityID)'
                                                ],
                                                'One way hashed authentication information'
                                            ),
                                            Table(
                                                'Person', 
                                                'PhoneNumberType', 
                                                [
                                                    TableField('PhoneNumberTypeID', 'SERIAL', ['PRIMARY KEY'], 'Primary key for telephone number type records.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Name of the telephone number type'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_PhoneNumberType_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Type of phone number of a person.'
                                            ),
                                            Table(
                                                'Person', 
                                                'PersonPhone', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL'], 'Business entity identification number. Foreign key to Person.BusinessEntityID.'), 
                                                    TableField('PhoneNumber', 'varchar(25)', ['NOT NULL'], 'Telephone number identification number.'), 
                                                    TableField('PhoneNumberTypeID', 'INT', ['NOT NULL'], 'Kind of phone number. Foreign key to PhoneNumberType.PhoneNumberTypeID.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_PersonPhone_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'Person', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES Person.Person(BusinessEntityID)'),
                                                    TableRelation('Person', 'PhoneNumberType', 'PhoneNumberTypeID', 'PhoneNumberTypeID', 'FOREIGN KEY (PhoneNumberTypeID) REFERENCES Person.PhoneNumberType(PhoneNumberTypeID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (BusinessEntityID, PhoneNumber, PhoneNumberTypeID)',
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES Person.Person(BusinessEntityID)',
                                                    'FOREIGN KEY (PhoneNumberTypeID) REFERENCES Person.PhoneNumberType(PhoneNumberTypeID)'
                                                ],
                                                'Telephone number and type of a person.'
                                            ),
                                            Table(
                                                'Person', 
                                                'CountryRegion', 
                                                [
                                                    TableField('CountryRegionCode', 'varchar(3)', ['NOT NULL', 'PRIMARY KEY'], 'ISO standard code for countries and regions.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Country or region name.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_CountryRegion_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Lookup table containing the ISO standard codes for countries and regions.'
                                            )
                                        ]
                                ),
                                Schema(
                                        'HumanResources',
                                        [
                                            Table(
                                                'HumanResources', 
                                                'Department', 
                                                [
                                                    TableField('DepartmentID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for Department records.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Name of the department.'), 
                                                    TableField('GroupName', 'varchar(50)', ['NOT NULL'], 'Name of the group to which the department belongs.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Department_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Lookup table containing the departments within the Adventure Works Cycles company.'
                                            ),
                                            Table(
                                                'HumanResources', 
                                                'Employee', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for Employee records. Foreign key to BusinessEntity.BusinessEntityID.'), 
                                                    TableField('NationalIDNumber', 'varchar(15)', ['NOT NULL'], 'Unique national identification number such as a social security number.'), 
                                                    TableField('LoginID', 'varchar(256)', ['NOT NULL'], 'Network login.'), 
                                                    TableField('OrganizationNode', 'VARCHAR', ['NOT NULL', 'DEFAULT \'/\''], 'Where the employee is located in corporate hierarchy.'), 
                                                    TableField('JobTitle', 'varchar(50)', ['NOT NULL'], 'Work title such as Buyer or Sales Representative.'), 
                                                    TableField('BirthDate', 'DATE', ['NOT NULL'], 'Date of birth.'), 
                                                    TableField('MaritalStatus', 'char(1)', ['NOT NULL'], 'M = Married, S = Single'), 
                                                    TableField('Gender', 'char(1)', ['NOT NULL'], 'M = Male, F = Female'), 
                                                    TableField('HireDate', 'DATE', ['NOT NULL'], 'Employee hired on this date.'), 
                                                    TableField('SalariedFlag', 'boolean', ['NOT NULL', 'CONSTRAINT "DF_Employee_SalariedFlag" DEFAULT (true)'], 'Job classification. 0 = Hourly, not exempt from collective bargaining. 1 = Salaried, exempt from collective bargaining.'), 
                                                    TableField('VacationHours', 'smallint', ['NOT NULL', 'CONSTRAINT "DF_Employee_VacationHours" DEFAULT (0)'], 'Number of available vacation hours.'), 
                                                    TableField('SickLeaveHours', 'smallint', ['NOT NULL', 'CONSTRAINT "DF_Employee_SickLeaveHours" DEFAULT (0)'], 'Number of available sick leave hours.'), 
                                                    TableField('CurrentFlag', 'boolean', ['NOT NULL', 'CONSTRAINT "DF_Employee_CurrentFlag" DEFAULT (true)'], '0 = Inactive, 1 = Active'), 
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_Employee_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Employee_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'Person', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES Person.Person(BusinessEntityID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_Employee_BirthDate" CHECK (BirthDate BETWEEN \'1930-01-01\' AND NOW() - INTERVAL \'18 years\')',
                                                    'CONSTRAINT "CK_Employee_MaritalStatus" CHECK (UPPER(MaritalStatus) IN (\'M\', \'S\'))',
                                                    'CONSTRAINT "CK_Employee_HireDate" CHECK (HireDate BETWEEN \'1996-07-01\' AND NOW() + INTERVAL \'1 day\')',
                                                    'CONSTRAINT "CK_Employee_Gender" CHECK (UPPER(Gender) IN (\'M\', \'F\'))',
                                                    'CONSTRAINT "CK_Employee_VacationHours" CHECK (VacationHours BETWEEN -40 AND 240)',
                                                    'CONSTRAINT "CK_Employee_SickLeaveHours" CHECK (SickLeaveHours BETWEEN 0 AND 120)',
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES Person.Person(BusinessEntityID)'
                                                ],
                                                'Employee information such as salary, department, and title.'
                                            ),
                                            Table(
                                                'HumanResources', 
                                                'EmployeeDepartmentHistory', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL'], 'Employee identification number. Foreign key to Employee.BusinessEntityID.'), 
                                                    TableField('DepartmentID', 'smallint', ['NOT NULL'], 'Department in which the employee worked including currently. Foreign key to Department.DepartmentID.'), 
                                                    TableField('ShiftID', 'smallint', ['NOT NULL'], 'Identifies which 8-hour shift the employee works. Foreign key to Shift.Shift.ID.'), 
                                                    TableField('StartDate', 'DATE', ['NOT NULL'], 'Date the employee started work in the department.'), 
                                                    TableField('EndDate', 'DATE', ['NULL'], 'Date the employee left the department. NULL = Current department.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_EmployeeDepartmentHistory_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('HumanResources', 'Department', 'DepartmentID', 'DepartmentID', 'FOREIGN KEY (DepartmentID) REFERENCES HumanResources.Department(DepartmentID)'),
                                                    TableRelation('HumanResources', 'Employee', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES HumanResources.Employee(BusinessEntityID)'),
                                                    TableRelation('HumanResources', 'Shift', 'ShiftID', 'ShiftID', 'FOREIGN KEY (ShiftID) REFERENCES HumanResources.Shift(ShiftID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_EmployeeDepartmentHistory_EndDate" CHECK ((EndDate >= StartDate) OR (EndDate IS NULL))',
                                                    'PRIMARY KEY (BusinessEntityID, StartDate, DepartmentID, ShiftID)',
                                                    'FOREIGN KEY (DepartmentID) REFERENCES HumanResources.Department(DepartmentID)',
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES HumanResources.Employee(BusinessEntityID)',
                                                    'FOREIGN KEY (ShiftID) REFERENCES HumanResources.Shift(ShiftID)'
                                                ],
                                                'Employee department transfers.'
                                            ),
                                            Table(
                                                'HumanResources', 
                                                'EmployeePayHistory', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL'], 'Employee identification number. Foreign key to Employee.BusinessEntityID.'), 
                                                    TableField('RateChangeDate', 'TIMESTAMP', ['NOT NULL'], 'Date the change in pay is effective'), 
                                                    TableField('Rate', 'numeric', ['NOT NULL'], 'Salary hourly rate.'), 
                                                    TableField('PayFrequency', 'smallint', ['NOT NULL'], '1 = Salary received monthly, 2 = Salary received biweekly'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_EmployeePayHistory_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('HumanResources', 'Employee', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES HumanResources.Employee(BusinessEntityID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_EmployeePayHistory_PayFrequency" CHECK (PayFrequency IN (1, 2))',
                                                    'CONSTRAINT "CK_EmployeePayHistory_Rate" CHECK (Rate BETWEEN 6.50 AND 200.00)',
                                                    'PRIMARY KEY (BusinessEntityID, RateChangeDate)',
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES HumanResources.Employee(BusinessEntityID)'
                                                ],
                                                'Employee pay history.'
                                            ),
                                            Table(
                                                'HumanResources', 
                                                'JobCandidate', 
                                                [
                                                    TableField('JobCandidateID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for JobCandidate records.'), 
                                                    TableField('BusinessEntityID', 'INT', ['NULL'], 'Employee identification number if applicant was hired. Foreign key to Employee.BusinessEntityID.'), 
                                                    TableField('Resume', 'XML', ['NULL'], 'Resume in XML format.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_JobCandidate_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('HumanResources', 'Employee', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES HumanResources.Employee(BusinessEntityID)')
                                                ],
                                                [
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES HumanResources.Employee(BusinessEntityID)'
                                                ],
                                                'Resumes submitted to Human Resources by job applicants.'
                                            ),
                                            Table(
                                                'HumanResources', 
                                                'Shift', 
                                                [
                                                    TableField('ShiftID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for Shift records.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Shift description.'), 
                                                    TableField('StartTime', 'time', ['NOT NULL'], 'Shift start time.'),
                                                    TableField('EndTime', 'time', ['NOT NULL'], 'Shift end time.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Shift_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Work shift lookup table.'
                                            )
                                        ]
                                    ),
                                    Schema(
                                        'Production',
                                        [
                                            Table(
                                                'Production', 
                                                'BillOfMaterials', 
                                                [
                                                    TableField('BillOfMaterialsID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for BillOfMaterials records.'), 
                                                    TableField('ProductAssemblyID', 'INT', ['NULL'], 'Parent product identification number. Foreign key to Product.ProductID.'), 
                                                    TableField('ComponentID', 'INT', ['NOT NULL'], 'Component identification number. Foreign key to Product.ProductID.'), 
                                                    TableField('StartDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_BillOfMaterials_StartDate" DEFAULT (NOW())'], 'Date the component started being used in the assembly item.'), 
                                                    TableField('EndDate', 'TIMESTAMP', ['NULL'], 'Date the component stopped being used in the assembly item.'), 
                                                    TableField('UnitMeasureCode', 'char(3)', ['NOT NULL'], 'Standard code identifying the unit of measure for the quantity.'), 
                                                    TableField('BOMLevel', 'smallint', ['NOT NULL'], 'Indicates the depth the component is from its parent (AssemblyID).'), 
                                                    TableField('PerAssemblyQty', 'decimal(8, 2)', ['NOT NULL', 'CONSTRAINT "DF_BillOfMaterials_PerAssemblyQty" DEFAULT (1.00)'], 'Quantity of the component needed to create the assembly.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_BillOfMaterials_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'Product', 'ProductAssemblyID', 'ProductID', 'FOREIGN KEY (ProductAssemblyID) REFERENCES Production.Product(ProductID)'),
                                                    TableRelation('Production', 'Product', 'ComponentID', 'ProductID', 'FOREIGN KEY (ComponentID) REFERENCES Production.Product(ProductID)'),
                                                    TableRelation('Production', 'UnitMeasure', 'UnitMeasureCode', 'UnitMeasureCode', 'FOREIGN KEY (UnitMeasureCode) REFERENCES Production.UnitMeasure(UnitMeasureCode)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_BillOfMaterials_EndDate" CHECK ((EndDate > StartDate) OR (EndDate IS NULL))',
                                                    'CONSTRAINT "CK_BillOfMaterials_ProductAssemblyID" CHECK (ProductAssemblyID <> ComponentID)',
                                                    'CONSTRAINT "CK_BillOfMaterials_BOMLevel" CHECK (((ProductAssemblyID IS NULL) AND (BOMLevel = 0) AND (PerAssemblyQty = 1.00)) OR ((ProductAssemblyID IS NOT NULL) AND (BOMLevel >= 1)))',
                                                    'CONSTRAINT "CK_BillOfMaterials_PerAssemblyQty" CHECK (PerAssemblyQty >= 1.00)',
                                                    'FOREIGN KEY (ProductAssemblyID) REFERENCES Production.Product(ProductID)',
                                                    'FOREIGN KEY (ComponentID) REFERENCES Production.Product(ProductID)',
                                                    'FOREIGN KEY (UnitMeasureCode) REFERENCES Production.UnitMeasure(UnitMeasureCode)',
                                                ],
                                                'Items required to make bicycles and bicycle subassemblies. It identifies the heirarchical relationship between a parent product and its components.'
                                            ),
                                            Table(
                                                'Production', 
                                                'Culture', 
                                                [
                                                    TableField('CultureID', 'char(6)', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for Culture records.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Culture description.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Culture_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Lookup table containing the languages in which some AdventureWorks data is stored.'
                                            ),
                                            Table(
                                                'Production', 
                                                'Document', 
                                                [
                                                    TableField('Title', 'varchar(50)', ['NOT NULL'], 'Title of the document.'),
                                                    TableField('Owner', 'INT', ['NOT NULL'], 'Employee who controls the document.  Foreign key to Employee.BusinessEntityID'),
                                                    TableField('FolderFlag', 'boolean', ['NOT NULL', 'CONSTRAINT "DF_Document_FolderFlag" DEFAULT (false)'], '0 = This is a folder, 1 = This is a document.'),
                                                    TableField('FileName', 'varchar(400)', ['NOT NULL'], 'File name of the document'),
                                                    TableField('FileExtension', 'varchar(8)', ['NULL'], 'File extension indicating the document type. For example, .doc or .txt.'),
                                                    TableField('Revision', 'char(5)', ['NOT NULL'], 'Revision number of the document.'),
                                                    TableField('ChangeNumber', 'INT', ['NOT NULL', 'CONSTRAINT "DF_Document_ChangeNumber" DEFAULT (0)'], 'Engineering change approval number.'),
                                                    TableField('Status', 'smallint', ['NOT NULL'], '1 = Pending approval, 2 = Approved, 3 = Obsolete'),
                                                    TableField('DocumentSummary', 'text', ['NULL'], 'Document abstract.'),
                                                    TableField('Document', 'bytea', ['NULL'], 'Complete document.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'UNIQUE', 'CONSTRAINT "DF_Document_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Document_ModifiedDate" DEFAULT (NOW())'], None),
                                                    TableField('DocumentNode', 'VARCHAR', ['DEFAULT \'/\'', 'PRIMARY KEY'], 'Primary key for Document records.')
                                                ],
                                                [
                                                    TableRelation('HumanResources', 'Employee', 'Owner', 'BusinessEntityID', 'FOREIGN KEY (Owner) REFERENCES HumanResources.Employee(BusinessEntityID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_Document_Status" CHECK (Status BETWEEN 1 AND 3)',
                                                    'FOREIGN KEY (Owner) REFERENCES HumanResources.Employee(BusinessEntityID)'
                                                ],
                                                'Product maintenance documents.'
                                            ),
                                            Table(
                                                'Production', 
                                                'ProductCategory', 
                                                [
                                                    TableField('ProductCategoryID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for ProductCategory records.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Category description.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_ProductCategory_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductCategory_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'High-level product categorization.'
                                            ),
                                            Table(
                                                'Production', 
                                                'ProductSubcategory', 
                                                [
                                                    TableField('ProductSubcategoryID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for ProductSubcategory records.'), 
                                                    TableField('ProductCategoryID', 'INT', ['NOT NULL'], 'Product category identification number. Foreign key to ProductCategory.ProductCategoryID.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Subcategory description.'), 
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_ProductSubcategory_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductSubcategory_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'ProductCategory', 'ProductCategoryID', 'ProductCategoryID', 'FOREIGN KEY (ProductCategoryID) REFERENCES Production.ProductCategory(ProductCategoryID)')
                                                ],
                                                [
                                                    'FOREIGN KEY (ProductCategoryID) REFERENCES Production.ProductCategory(ProductCategoryID)'
                                                ],
                                                'Product subcategories. See ProductCategory table.'
                                            ),
                                            Table(
                                                'Production', 
                                                'ProductModel', 
                                                [
                                                    TableField('ProductModelID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for ProductModel records.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Product model description.'), 
                                                    TableField('CatalogDescription', 'XML', ['NULL'], 'Detailed product catalog information in xml format.'), 
                                                    TableField('Instructions', 'XML', ['NULL'], 'Manufacturing instructions in xml format.'), 
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_ProductModel_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductModel_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Product model classification.'
                                            ),
                                            Table(
                                                'Production', 
                                                'Product', 
                                                [
                                                    TableField('ProductID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for Product records.'),
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Name of the product.'),
                                                    TableField('ProductNumber', 'varchar(25)', ['NOT NULL'], 'Unique product identification number.'),
                                                    TableField('MakeFlag', 'boolean', ['NOT NULL', 'CONSTRAINT "DF_Product_MakeFlag" DEFAULT (true)'], '0 = Product is purchased, 1 = Product is manufactured in-house.'),
                                                    TableField('FinishedGoodsFlag', 'boolean', ['NOT NULL', 'CONSTRAINT "DF_Product_FinishedGoodsFlag" DEFAULT (true)'], '0 = Product is not a salable item. 1 = Product is salable.'),
                                                    TableField('Color', 'varchar(15)', ['NULL'], 'Product color.'),
                                                    TableField('SafetyStockLevel', 'smallint', ['NOT NULL'], 'Minimum inventory quantity.'),
                                                    TableField('ReorderPoint', 'smallint', ['NOT NULL'], 'Inventory level that triggers a purchase order or work order.'),
                                                    TableField('StandardCost', 'numeric', ['NOT NULL'], 'Standard cost of the product.'),
                                                    TableField('ListPrice', 'numeric', ['NOT NULL'], 'Selling price.'),
                                                    TableField('Size', 'varchar(5)', ['NULL'], 'Product size.'),
                                                    TableField('SizeUnitMeasureCode', 'char(3)', ['NULL'], 'Unit of measure for Size column.'),
                                                    TableField('WeightUnitMeasureCode', 'char(3)', ['NULL'], 'Unit of measure for Weight column.'),
                                                    TableField('Weight', 'decimal(8, 2)', ['NULL'], 'Product weight.'),
                                                    TableField('DaysToManufacture', 'INT', ['NOT NULL'], 'Number of days required to manufacture the product.'),
                                                    TableField('ProductLine', 'char(2)', ['NULL'], 'R = Road, M = Mountain, T = Touring, S = Standard'),
                                                    TableField('Class', 'char(2)', ['NULL'], 'H = High, M = Medium, L = Low'),
                                                    TableField('Style', 'char(2)', ['NULL'], 'W = Womens, M = Mens, U = Universal'),
                                                    TableField('ProductSubcategoryID', 'INT', ['NULL'], 'Product is a member of this product subcategory. Foreign key to ProductSubCategory.ProductSubCategoryID.'),
                                                    TableField('ProductModelID', 'INT', ['NULL'], 'Product is a member of this product model. Foreign key to ProductModel.ProductModelID.'),
                                                    TableField('SellStartDate', 'TIMESTAMP', ['NOT NULL'], 'Date the product was available for sale.'),
                                                    TableField('SellEndDate', 'TIMESTAMP', ['NULL'], 'Date the product was no longer available for sale.'),
                                                    TableField('DiscontinuedDate', 'TIMESTAMP', ['NULL'], 'Date the product was discontinued.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_Product_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Product_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'UnitMeasure', 'SizeUnitMeasureCode', 'UnitMeasureCode', 'FOREIGN KEY (SizeUnitMeasureCode) REFERENCES Production.UnitMeasure(UnitMeasureCode)'),
                                                    TableRelation('Production', 'UnitMeasure', 'WeightUnitMeasureCode', 'UnitMeasureCode', 'FOREIGN KEY (WeightUnitMeasureCode) REFERENCES Production.UnitMeasure(UnitMeasureCode)'),
                                                    TableRelation('Production', 'ProductModel', 'ProductModelID', 'ProductModelID', 'FOREIGN KEY (ProductModelID) REFERENCES Production.ProductModel(ProductModelID)'),
                                                    TableRelation('Production', 'ProductSubcategory', 'ProductSubcategoryID', 'ProductSubcategoryID', 'FOREIGN KEY (ProductSubcategoryID) REFERENCES Production.ProductSubcategory(ProductSubcategoryID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_Product_SafetyStockLevel" CHECK (SafetyStockLevel > 0)',
                                                    'CONSTRAINT "CK_Product_ReorderPoint" CHECK (ReorderPoint > 0)',
                                                    'CONSTRAINT "CK_Product_StandardCost" CHECK (StandardCost >= 0.00)',
                                                    'CONSTRAINT "CK_Product_ListPrice" CHECK (ListPrice >= 0.00)',
                                                    'CONSTRAINT "CK_Product_Weight" CHECK (Weight > 0.00 OR Weight IS NULL)',
                                                    'CONSTRAINT "CK_Product_DaysToManufacture" CHECK (DaysToManufacture >= 0)',
                                                    'CONSTRAINT "CK_Product_ProductLine" CHECK(UPPER(ProductLine) IN (\'S\', \'T\', \'M\', \'R\') OR ProductLine IS NULL)',
                                                    'CONSTRAINT "CK_Product_Class" CHECK (UPPER(Class) IN (\'L\', \'M\', \'H\') OR Class IS NULL)',
                                                    'CONSTRAINT "CK_Product_Style" CHECK (UPPER(Style) IN (\'W\', \'M\', \'U\') OR Style IS NULL)',
                                                    'CONSTRAINT "CK_Product_SellEndDate" CHECK ((SellEndDate >= SellStartDate) OR (SellEndDate IS NULL))',
                                                    'FOREIGN KEY (SizeUnitMeasureCode) REFERENCES Production.UnitMeasure(UnitMeasureCode)',
                                                    'FOREIGN KEY (WeightUnitMeasureCode) REFERENCES Production.UnitMeasure(UnitMeasureCode)',
                                                    'FOREIGN KEY (ProductModelID) REFERENCES Production.ProductModel(ProductModelID)',
                                                    'FOREIGN KEY (ProductSubcategoryID) REFERENCES Production.ProductSubcategory(ProductSubcategoryID)'
                                                ],
                                                'Products sold or used in the manufacturing of sold products.'
                                            ),
                                            Table(
                                                'Production', 
                                                'ProductCostHistory', 
                                                [
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Product identification number. Foreign key to Product.ProductID'),
                                                    TableField('StartDate', 'TIMESTAMP', ['NOT NULL'], 'Product cost start date.'),
                                                    TableField('EndDate', 'TIMESTAMP', ['NULL'], 'Product cost end date.'),
                                                    TableField('StandardCost', 'numeric', ['NOT NULL'], 'Standard cost of the product.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductCostHistory_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'Product', 'ProductID', 'ProductID', 'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (ProductID, StartDate)',
                                                    'CONSTRAINT "CK_ProductCostHistory_EndDate" CHECK ((EndDate >= StartDate) OR (EndDate IS NULL))',
                                                    'CONSTRAINT "CK_ProductCostHistory_StandardCost" CHECK (StandardCost >= 0.00)',
                                                    'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)'
                                                ],
                                                'Changes in the cost of a product over time.'
                                            ),
                                            Table(
                                                'Production', 
                                                'ProductDescription', 
                                                [
                                                    TableField('ProductDescriptionID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for ProductDescription records.'), 
                                                    TableField('Description', 'varchar(400)', ['NOT NULL'], 'Description of the product.'), 
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_ProductDescription_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductDescription_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Product descriptions in several languages.'
                                            ),
                                            Table(
                                                'Production', 
                                                'ProductDocument', 
                                                [
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Product identification number. Foreign key to Product.ProductID.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductDocument_ModifiedDate" DEFAULT (NOW())'], None),
                                                    TableField('DocumentNode', 'VARCHAR', ['DEFAULT \'/\''], 'Document identification number. Foreign key to Document.DocumentNode.')
                                                ],
                                                [
                                                    TableRelation('Production', 'Product', 'ProductID', 'ProductID', 'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)'),
                                                    TableRelation('Production', 'Document', 'DocumentNode', 'DocumentNode', 'FOREIGN KEY (DocumentNode) REFERENCES Production.Document(DocumentNode)')
                                                ],
                                                [
                                                   'PRIMARY KEY (ProductID, DocumentNode)',
                                                   'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)',
                                                   'FOREIGN KEY (DocumentNode) REFERENCES Production.Document(DocumentNode)'
                                                ],
                                                'Cross-reference table mapping products to related product documents.'
                                            ),
                                            Table(
                                                'Production', 
                                                'Location', 
                                                [
                                                    TableField('LocationID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for Location records.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Location description.'), 
                                                    TableField('CostRate', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_Location_CostRate" DEFAULT (0.00)'], 'Standard hourly cost of the manufacturing location.'),
                                                    TableField('Availability', 'decimal(8, 2)', ['NOT NULL', 'CONSTRAINT "DF_Location_Availability" DEFAULT (0.00)'], 'Work capacity (in hours) of the manufacturing location.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Location_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                [
                                                    'CONSTRAINT "CK_Location_CostRate" CHECK (CostRate >= 0.00)',
                                                    'CONSTRAINT "CK_Location_Availability" CHECK (Availability >= 0.00)'
                                                ],
                                                'Product inventory and manufacturing locations.'
                                            ),
                                            Table(
                                                'Production', 
                                                'ProductInventory', 
                                                [
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Product identification number. Foreign key to Product.ProductID.'),
                                                    TableField('LocationID', 'smallint', ['NOT NULL'], 'Inventory location identification number. Foreign key to Location.LocationID.'),
                                                    TableField('Shelf', 'varchar(10)', ['NOT NULL'], 'Storage compartment within an inventory location.'),
                                                    TableField('Bin', 'smallint', ['NOT NULL'], 'Storage container on a shelf in an inventory location.'),
                                                    TableField('Quantity', 'smallint', ['NOT NULL', 'CONSTRAINT "DF_ProductInventory_Quantity" DEFAULT (0)'], 'Quantity of products in the inventory location.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_ProductInventory_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductInventory_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'Location', 'LocationID', 'LocationID', 'FOREIGN KEY (LocationID) REFERENCES Production.Location(LocationID)'),
                                                    TableRelation('Production', 'Product', 'ProductID', 'ProductID', 'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (ProductID, LocationID)',
                                                    'FOREIGN KEY (LocationID) REFERENCES Production.Location(LocationID)',
                                                    'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)',
                                                    'CONSTRAINT "CK_ProductInventory_Bin" CHECK (Bin BETWEEN 0 AND 100)'
                                                ],
                                                'Product inventory information.'
                                            ),
                                            Table(
                                                'Production', 
                                                'ProductListPriceHistory', 
                                                [
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Product identification number. Foreign key to Product.ProductID.'),
                                                    TableField('StartDate', 'TIMESTAMP', ['NOT NULL'], 'List price start date.'),
                                                    TableField('EndDate', 'TIMESTAMP', ['NULL'], 'List price end date.'),
                                                    TableField('ListPrice', 'numeric', ['NOT NULL'], 'Product list price.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductListPriceHistory_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'Product', 'ProductID', 'ProductID', 'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (ProductID, StartDate)',
                                                    'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)',
                                                    'CONSTRAINT "CK_ProductListPriceHistory_EndDate" CHECK ((EndDate >= StartDate) OR (EndDate IS NULL))',
                                                    'CONSTRAINT "CK_ProductListPriceHistory_ListPrice" CHECK (ListPrice > 0.00)'
                                                ],
                                                'Changes in the list price of a product over time.'
                                            ),
                                            Table(
                                                'Production', 
                                                'Illustration', 
                                                [
                                                    TableField('IllustrationID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for Illustration records.'), 
                                                    TableField('Diagram', 'XML', ['NULL'], 'Illustrations used in manufacturing instructions. Stored as XML.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Illustration_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Bicycle assembly diagrams.'
                                            ),
                                            Table(
                                                'Production', 
                                                'ProductModelIllustration', 
                                                [
                                                    TableField('ProductModelID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to ProductModel.ProductModelID.'),
                                                    TableField('IllustrationID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to Illustration.IllustrationID.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductModelIllustration_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'ProductModel', 'ProductModelID', 'ProductModelID', 'FOREIGN KEY (ProductModelID) REFERENCES Production.ProductModel(ProductModelID)'),
                                                    TableRelation('Production', 'Illustration', 'IllustrationID', 'IllustrationID', 'FOREIGN KEY (IllustrationID) REFERENCES Production.Illustration(IllustrationID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (ProductModelID, IllustrationID)',
                                                    'FOREIGN KEY (ProductModelID) REFERENCES Production.ProductModel(ProductModelID)',
                                                    'FOREIGN KEY (IllustrationID) REFERENCES Production.Illustration(IllustrationID)'
                                                ],
                                                'Cross-reference table mapping product models and illustrations.'
                                            ),
                                            Table(
                                                'Production', 
                                                'ProductModelProductDescriptionCulture', 
                                                [
                                                    TableField('ProductModelID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to ProductModel.ProductModelID.'),
                                                    TableField('ProductDescriptionID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to ProductDescription.ProductDescriptionID.'),
                                                    TableField('CultureID', 'char(6)', ['NOT NULL'], 'Culture identification number. Foreign key to Culture.CultureID.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductModelProductDescriptionCulture_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'ProductDescription', 'ProductDescriptionID', 'ProductDescriptionID', 'FOREIGN KEY (ProductDescriptionID) REFERENCES Production.ProductDescription(ProductDescriptionID)'),
                                                    TableRelation('Production', 'Culture', 'CultureID', 'CultureID', 'FOREIGN KEY (CultureID) REFERENCES Production.Culture(CultureID)'),
                                                    TableRelation('Production', 'ProductModel', 'ProductModelID', 'ProductModelID', 'FOREIGN KEY (ProductModelID) REFERENCES Production.ProductModel(ProductModelID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (ProductModelID, ProductDescriptionID, CultureID)',
                                                    'FOREIGN KEY (ProductDescriptionID) REFERENCES Production.ProductDescription(ProductDescriptionID)',
                                                    'FOREIGN KEY (CultureID) REFERENCES Production.Culture(CultureID)',
                                                    'FOREIGN KEY (ProductModelID) REFERENCES Production.ProductModel(ProductModelID)'
                                                ],
                                                'Cross-reference table mapping product descriptions and the language the description is written in.'
                                            ),
                                            Table(
                                                'Production', 
                                                'ProductPhoto', 
                                                [
                                                    TableField('ProductPhotoID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for ProductPhoto records.'), 
                                                    TableField('ThumbNailPhoto', 'bytea', ['NULL'], 'Small image of the product.'),
                                                    TableField('ThumbnailPhotoFileName', 'varchar(50)', ['NULL'], 'Small image file name.'),
                                                    TableField('LargePhoto', 'bytea', ['NULL'], 'Large image of the product.'),
                                                    TableField('LargePhotoFileName', 'varchar(50)', ['NULL'], 'Large image file name.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductPhoto_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Product images.'
                                            ),
                                            Table(
                                                'Production',
                                                'ProductProductPhoto',
                                                [
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Product identification number. Foreign key to Product.ProductID.'),
                                                    TableField('ProductPhotoID', 'INT', ['NOT NULL'], 'Product photo identification number. Foreign key to ProductPhoto.ProductPhotoID.'),
                                                    TableField('"primary"', 'boolean', ['NOT NULL', 'CONSTRAINT "DF_ProductProductPhoto_Primary" DEFAULT (false)'], '0 = Photo is not the principal image. 1 = Photo is the principal image.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductProductPhoto_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'Product', 'ProductID', 'ProductID', 'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)'),
                                                    TableRelation('Production', 'ProductPhoto', 'ProductPhotoID', 'ProductPhotoID', 'FOREIGN KEY (ProductPhotoID) REFERENCES Production.ProductPhoto(ProductPhotoID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (ProductID, ProductPhotoID)',
                                                    'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)',
                                                    'FOREIGN KEY (ProductPhotoID) REFERENCES Production.ProductPhoto(ProductPhotoID)'
                                                ],
                                                'Cross-reference table mapping products and product photos.'
                                            ),
                                            Table(
                                                'Production',
                                                'ProductReview',
                                                [
                                                    TableField('ProductReviewID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for ProductReview records.'),
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Product identification number. Foreign key to Product.ProductID.'),
                                                    TableField('ReviewerName', 'varchar(50)', ['NOT NULL'], 'Name of the reviewer.'),
                                                    TableField('ReviewDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductReview_ReviewDate" DEFAULT (NOW())'], 'Date review was submitted.'),
                                                    TableField('EmailAddress', 'varchar(50)', ['NOT NULL'], "Reviewer's e-mail address."),
                                                    TableField('Rating', 'INT', ['NOT NULL'], 'Product rating given by the reviewer. Scale is 1 to 5 with 5 as the highest rating.'),
                                                    TableField('Comments', 'varchar(3850)', [], "Reviewer's comments"),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductReview_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'Product', 'ProductID', 'ProductID', 'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)')
                                                ],
                                                [
                                                    'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)',
                                                    'CONSTRAINT "CK_ProductReview_Rating" CHECK (Rating BETWEEN 1 AND 5)'
                                                ],
                                                'Customer reviews of products they have purchased.'
                                            ),
                                            Table(
                                                'Production',
                                                'ScrapReason',
                                                [
                                                    TableField('ScrapReasonID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for ScrapReason records.'),
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Failure description.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ScrapReason_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Manufacturing failure reasons lookup table.'
                                            ),
                                            Table(
                                                'Production',
                                                'TransactionHistory',
                                                [
                                                    TableField('TransactionID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for TransactionHistory records.'),
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Product identification number. Foreign key to Product.ProductID.'),
                                                    TableField('ReferenceOrderID', 'INT', ['NOT NULL'], 'Purchase order, sales order, or work order identification number.'),
                                                    TableField('ReferenceOrderLineID', 'INT', ['NOT NULL', 'CONSTRAINT "DF_TransactionHistory_ReferenceOrderLineID" DEFAULT (0)'], 'Line number associated with the purchase order, sales order, or work order.'),
                                                    TableField('TransactionDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_TransactionHistory_TransactionDate" DEFAULT (NOW())'], 'Date and time of the transaction.'),
                                                    TableField('TransactionType', 'char(1)', ['NOT NULL'], "W = WorkOrder, S = SalesOrder, P = PurchaseOrder."),
                                                    TableField('Quantity', 'INT', ['NOT NULL'], 'Product quantity.'),
                                                    TableField('ActualCost', 'numeric', ['NOT NULL'], 'Product cost.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_TransactionHistory_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'Product', 'ProductID', 'ProductID', 'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)')
                                                ],
                                                [
                                                    'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)',
                                                    'CONSTRAINT "CK_TransactionHistory_TransactionType" CHECK (UPPER(TransactionType) IN (\'W\', \'S\', \'P\'))'
                                                ],
                                                'Record of each purchase order, sales order, or work order transaction year to date.'
                                            ),
                                            Table(
                                                'Production', 
                                                'TransactionHistoryArchive', 
                                                [
                                                    TableField('TransactionID', 'INT', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for TransactionHistoryArchive records.'), 
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Product identification number. Foreign key to Product.ProductID.'), 
                                                    TableField('ReferenceOrderID', 'INT', ['NOT NULL'], 'Purchase order, sales order, or work order identification number.'), 
                                                    TableField('ReferenceOrderLineID', 'INT', ['NOT NULL', 'CONSTRAINT "DF_TransactionHistoryArchive_ReferenceOrderLineID" DEFAULT (0)'], 'Line number associated with the purchase order, sales order, or work order.'), 
                                                    TableField('TransactionDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_TransactionHistoryArchive_TransactionDate" DEFAULT (NOW())'], 'Date and time of the transaction.'), 
                                                    TableField('TransactionType', 'char(1)', ['NOT NULL'], 'W = Work Order, S = Sales Order, P = Purchase Order'), 
                                                    TableField('Quantity', 'INT', ['NOT NULL'], 'Product quantity.'), 
                                                    TableField('ActualCost', 'numeric', ['NOT NULL'], 'Product cost.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_TransactionHistoryArchive_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                [
                                                    'CONSTRAINT "CK_TransactionHistoryArchive_TransactionType" CHECK (UPPER(TransactionType) IN (\'W\', \'S\', \'P\'))'
                                                ],
                                                'Transactions for previous years.'
                                            ),
                                            Table(
                                                'Production',
                                                'UnitMeasure',
                                                [
                                                    TableField('UnitMeasureCode', 'char(3)', ['NOT NULL', 'PRIMARY KEY'], 'Primary key.'),
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Unit of measure description.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_UnitMeasure_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Unit of measure lookup table.'
                                            ),
                                            Table(
                                                'Production',
                                                'WorkOrder',
                                                [
                                                    TableField('WorkOrderID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for WorkOrder records.'),
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Product identification number. Foreign key to Product.ProductID.'),
                                                    TableField('OrderQty', 'INT', ['NOT NULL'], 'Product quantity to build.'),
                                                    TableField('ScrappedQty', 'smallint', ['NOT NULL'], 'Quantity that failed inspection.'),
                                                    TableField('StartDate', 'TIMESTAMP', ['NOT NULL'], 'Work order start date.'),
                                                    TableField('EndDate', 'TIMESTAMP', ['NULL'], 'Work order end date.'),
                                                    TableField('DueDate', 'TIMESTAMP', ['NOT NULL'], 'Work order due date.'),
                                                    TableField('ScrapReasonID', 'smallint', ['NULL'], 'Reason for inspection failure.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_WorkOrder_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'Product', 'ProductID', 'ProductID', 'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)'),
                                                    TableRelation('Production', 'ScrapReason', 'ScrapReasonID', 'ScrapReasonID', 'FOREIGN KEY (ScrapReasonID) REFERENCES Production.ScrapReason(ScrapReasonID)')
                                                ],
                                                [
                                                    'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)',
                                                    'FOREIGN KEY (ScrapReasonID) REFERENCES Production.ScrapReason(ScrapReasonID)',
                                                    'CONSTRAINT "CK_WorkOrder_OrderQty" CHECK (OrderQty > 0)',
                                                    'CONSTRAINT "CK_WorkOrder_ScrappedQty" CHECK (ScrappedQty >= 0)',
                                                    'CONSTRAINT "CK_WorkOrder_EndDate" CHECK ((EndDate >= StartDate) OR (EndDate IS NULL))'
                                                ],
                                                'Manufacturing work orders.'
                                            ),
                                            Table(
                                                'Production',
                                                'WorkOrderRouting',
                                                [
                                                    TableField('WorkOrderID', 'INT', ['NOT NULL'], 'Primary key for WorkOrderRouting records. Foreign key to WorkOrder.WorkOrderID.'),
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Primary key for WorkOrderRouting records'),
                                                    TableField('OperationSequence', 'smallint', ['NOT NULL'], 'Primary key for WorkOrderRouting records'),
                                                    TableField('LocationID', 'smallint', ['NOT NULL'], 'Location of the Work Order. Foreign key to Location.LocationID.'),
                                                    TableField('ScheduledStartDate', 'TIMESTAMP', ['NOT NULL'], 'Scheduled start date for the manufacturing work order.'),
                                                    TableField('ScheduledEndDate', 'TIMESTAMP', ['NOT NULL'], 'Scheduled end date for the manufacturing work order.'),
                                                    TableField('ActualStartDate', 'TIMESTAMP', ['NULL'], 'Actual start date for the manufacturing work order.'),
                                                    TableField('ActualEndDate', 'TIMESTAMP', ['NULL'], 'Actual end date for the manufacturing work order.'),
                                                    TableField('ActualResourceHrs', 'decimal(9, 4)', ['NULL'], 'Work hours required for the manufacturing work order.'),
                                                    TableField('PlannedCost', 'numeric', ['NOT NULL'], 'Estimated cost for the manufacturing work order.'),
                                                    TableField('ActualCost', 'numeric', ['NULL'], 'Actual cost for the manufacturing work order.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_WorkOrderRouting_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'Location', 'LocationID', 'LocationID', 'FOREIGN KEY (LocationID) REFERENCES Production.Location(LocationID)'),
                                                    TableRelation('Production', 'WorkOrder', 'WorkOrderID', 'WorkOrderID', 'FOREIGN KEY (WorkOrderID) REFERENCES Production.WorkOrder(WorkOrderID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (WorkOrderID, ProductID, OperationSequence)',
                                                    'FOREIGN KEY (LocationID) REFERENCES Production.Location(LocationID)',
                                                    'FOREIGN KEY (WorkOrderID) REFERENCES Production.WorkOrder(WorkOrderID)',
                                                    'CONSTRAINT "CK_WorkOrderRouting_ScheduledEndDate" CHECK (ScheduledEndDate >= ScheduledStartDate)',
                                                    'CONSTRAINT "CK_WorkOrderRouting_ActualEndDate" CHECK ((ActualEndDate >= ActualStartDate) OR (ActualEndDate IS NULL) OR (ActualStartDate IS NULL))',
                                                    'CONSTRAINT "CK_WorkOrderRouting_ActualResourceHrs" CHECK (ActualResourceHrs >= 0.0000)',
                                                    'CONSTRAINT "CK_WorkOrderRouting_PlannedCost" CHECK (PlannedCost > 0.00)',
                                                    'CONSTRAINT "CK_WorkOrderRouting_ActualCost" CHECK (ActualCost > 0.00)'
                                                ],
                                                'Information about the schedule and costs of manufacturing work orders'
                                            )
                                        ]
                                    ),
                                    Schema(
                                        'Purchasing',
                                        [
                                            Table(
                                                'Purchasing',
                                                'ProductVendor',
                                                [
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to Product.ProductID.'),
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to Vendor.BusinessEntityID.'),
                                                    TableField('AverageLeadTime', 'INT', ['NOT NULL'], 'The average span of time (in days) between placing an order with the vendor and receiving the purchased product.'),
                                                    TableField('StandardPrice', 'numeric', ['NOT NULL'], "The vendor's usual selling price."),
                                                    TableField('LastReceiptCost', 'numeric', ['NULL'], 'The selling price when last purchased.'),
                                                    TableField('LastReceiptDate', 'TIMESTAMP', ['NULL'], 'Date the product was last received by the vendor.'),
                                                    TableField('MinOrderQty', 'INT', ['NOT NULL'], 'The maximum quantity that should be ordered.'),
                                                    TableField('MaxOrderQty', 'INT', ['NOT NULL'], 'The minimum quantity that should be ordered.'),
                                                    TableField('OnOrderQty', 'INT', ['NULL'], 'The quantity currently on order.'),
                                                    TableField('UnitMeasureCode', 'char(3)', ['NOT NULL'], "The product's unit of measure."),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ProductVendor_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'Product', 'ProductID', 'ProductID', 'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)'),
                                                    TableRelation('Production', 'UnitMeasure', 'UnitMeasureCode', 'UnitMeasureCode', 'FOREIGN KEY (UnitMeasureCode) REFERENCES Production.UnitMeasure(UnitMeasureCode)'),
                                                    TableRelation('Purchasing', 'Vendor', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES Purchasing.Vendor(BusinessEntityID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (ProductID, BusinessEntityID)',
                                                    'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)',
                                                    'FOREIGN KEY (UnitMeasureCode) REFERENCES Production.UnitMeasure(UnitMeasureCode)',
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES Purchasing.Vendor(BusinessEntityID)',
                                                    'CONSTRAINT "CK_ProductVendor_AverageLeadTime" CHECK (AverageLeadTime >= 1)',
                                                    'CONSTRAINT "CK_ProductVendor_StandardPrice" CHECK (StandardPrice > 0.00)',
                                                    'CONSTRAINT "CK_ProductVendor_LastReceiptCost" CHECK (LastReceiptCost > 0.00)',
                                                    'CONSTRAINT "CK_ProductVendor_MinOrderQty" CHECK (MinOrderQty >= 1)',
                                                    'CONSTRAINT "CK_ProductVendor_MaxOrderQty" CHECK (MaxOrderQty >= 1)',
                                                    'CONSTRAINT "CK_ProductVendor_OnOrderQty" CHECK (OnOrderQty >= 0)'
                                                ],
                                                'Cross-reference table mapping vendors with the products they supply.'
                                            ),
                                            Table(
                                                'Purchasing', 
                                                'PurchaseOrderDetail', 
                                                [
                                                    TableField('PurchaseOrderID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to PurchaseOrderHeader.PurchaseOrderID.'),
                                                    TableField('PurchaseOrderDetailID', 'SERIAL', ['NOT NULL'], 'Primary key. One line number per purchased product.'),
                                                    TableField('DueDate', 'TIMESTAMP', ['NOT NULL'], 'Date the product is expected to be received.'),
                                                    TableField('OrderQty', 'smallint', ['NOT NULL'], 'Quantity ordered.'),
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Product identification number. Foreign key to Product.ProductID.'),
                                                    TableField('UnitPrice', 'numeric', ['NOT NULL'], "Vendor's selling price of a single product."),
                                                    TableField('ReceivedQty', 'decimal(8, 2)', ['NOT NULL'], 'Quantity actually received from the vendor.'),
                                                    TableField('RejectedQty', 'decimal(8, 2)', ['NOT NULL'], 'Quantity rejected during inspection.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_PurchaseOrderDetail_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'Product', 'ProductID', 'ProductID', 'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)'),
                                                    TableRelation('Purchasing', 'PurchaseOrderHeader', 'PurchaseOrderID', 'PurchaseOrderID', 'FOREIGN KEY (PurchaseOrderID) REFERENCES Purchasing.PurchaseOrderHeader(PurchaseOrderID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (PurchaseOrderID, PurchaseOrderDetailID)',
                                                    'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)',
                                                    'FOREIGN KEY (PurchaseOrderID) REFERENCES Purchasing.PurchaseOrderHeader(PurchaseOrderID)',
                                                    'CONSTRAINT "CK_PurchaseOrderDetail_OrderQty" CHECK (OrderQty > 0)',
                                                    'CONSTRAINT "CK_PurchaseOrderDetail_UnitPrice" CHECK (UnitPrice >= 0.00)',
                                                    'CONSTRAINT "CK_PurchaseOrderDetail_ReceivedQty" CHECK (ReceivedQty >= 0.00)',
                                                    'CONSTRAINT "CK_PurchaseOrderDetail_RejectedQty" CHECK (RejectedQty >= 0.00)'
                                                ],
                                                'Individual products associated with a specific purchase order. See PurchaseOrderHeader.'
                                            ),
                                            Table(
                                                'Purchasing',
                                                'PurchaseOrderHeader',
                                                [
                                                    TableField('PurchaseOrderID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key.'),
                                                    TableField('RevisionNumber', 'smallint', ['NOT NULL', 'CONSTRAINT "DF_PurchaseOrderHeader_RevisionNumber" DEFAULT (0)'], 'Incremental number to track changes to the purchase order over time.'),
                                                    TableField('Status', 'smallint', ['NOT NULL', 'CONSTRAINT "DF_PurchaseOrderHeader_Status" DEFAULT (1)'], 'Order current status. 1 = Pending; 2 = Approved; 3 = Rejected; 4 = Complete'),
                                                    TableField('EmployeeID', 'INT', ['NOT NULL'], 'Employee who created the purchase order. Foreign key to Employee.BusinessEntityID.'),
                                                    TableField('VendorID', 'INT', ['NOT NULL'], 'Vendor with whom the purchase order is placed. Foreign key to Vendor.BusinessEntityID.'),
                                                    TableField('ShipMethodID', 'INT', ['NOT NULL'], 'Shipping method. Foreign key to ShipMethod.ShipMethodID.'),
                                                    TableField('OrderDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_PurchaseOrderHeader_OrderDate" DEFAULT (NOW())'], 'Purchase order creation date.'),
                                                    TableField('ShipDate', 'TIMESTAMP', ['NULL'], 'Estimated shipment date from the vendor.'),
                                                    TableField('SubTotal', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_PurchaseOrderHeader_SubTotal" DEFAULT (0.00)'], 'Purchase order subtotal. Computed as SUM(PurchaseOrderDetail.LineTotal) for the appropriate PurchaseOrderID.'),
                                                    TableField('TaxAmt', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_PurchaseOrderHeader_TaxAmt" DEFAULT (0.00)'], 'Tax amount.'),
                                                    TableField('Freight', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_PurchaseOrderHeader_Freight" DEFAULT (0.00)'], 'Shipping cost.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_PurchaseOrderHeader_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('HumanResources', 'Employee', 'EmployeeID', 'BusinessEntityID', 'FOREIGN KEY (EmployeeID) REFERENCES HumanResources.Employee(BusinessEntityID)'),
                                                    TableRelation('Purchasing', 'Vendor', 'VendorID', 'BusinessEntityID', 'FOREIGN KEY (VendorID) REFERENCES Purchasing.Vendor(BusinessEntityID)'),
                                                    TableRelation('Purchasing', 'ShipMethod', 'ShipMethodID', 'ShipMethodID', 'FOREIGN KEY (ShipMethodID) REFERENCES Purchasing.ShipMethod(ShipMethodID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_PurchaseOrderHeader_Status" CHECK (Status BETWEEN 1 AND 4)',
                                                    'CONSTRAINT "CK_PurchaseOrderHeader_ShipDate" CHECK ((ShipDate >= OrderDate) OR (ShipDate IS NULL))',
                                                    'CONSTRAINT "CK_PurchaseOrderHeader_SubTotal" CHECK (SubTotal >= 0.00)',
                                                    'CONSTRAINT "CK_PurchaseOrderHeader_TaxAmt" CHECK (TaxAmt >= 0.00)',
                                                    'CONSTRAINT "CK_PurchaseOrderHeader_Freight" CHECK (Freight >= 0.00)',
                                                    'FOREIGN KEY (EmployeeID) REFERENCES HumanResources.Employee(BusinessEntityID)',
                                                    'FOREIGN KEY (VendorID) REFERENCES Purchasing.Vendor(BusinessEntityID)',
                                                    'FOREIGN KEY (ShipMethodID) REFERENCES Purchasing.ShipMethod(ShipMethodID)'
                                                ],
                                                'General purchase order information. See PurchaseOrderDetail.'
                                            ),
                                            Table(
                                                'Purchasing', 
                                                'ShipMethod', 
                                                [
                                                    TableField('ShipMethodID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for ShipMethod records.'),
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Shipping company name.'),
                                                    TableField('ShipBase', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_ShipMethod_ShipBase" DEFAULT (0.00)'], 'Minimum shipping charge.'),
                                                    TableField('ShipRate', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_ShipMethod_ShipRate" DEFAULT (0.00)'], 'Shipping charge per pound.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_ShipMethod_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ShipMethod_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                [
                                                    'CONSTRAINT "CK_ShipMethod_ShipBase" CHECK (ShipBase > 0.00)',
                                                    'CONSTRAINT "CK_ShipMethod_ShipRate" CHECK (ShipRate > 0.00)'
                                                ],
                                                'Shipping company lookup table.'
                                            ),
                                            Table(
                                                'Purchasing', 
                                                'Vendor', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for Vendor records.  Foreign key to BusinessEntity.BusinessEntityID'), 
                                                    TableField('AccountNumber', 'varchar(15)', ['NOT NULL'], 'Vendor account (identification) number.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Company name.'), 
                                                    TableField('CreditRating', 'smallint', ['NOT NULL'], '1 = Superior, 2 = Excellent, 3 = Above average, 4 = Average, 5 = Below average'), 
                                                    TableField('PreferredVendorStatus', 'boolean', ['NOT NULL', 'CONSTRAINT "DF_Vendor_PreferredVendorStatus" DEFAULT (true)'], '0 = Do not use if another vendor is available. 1 = Preferred over other vendors supplying the same product.'), 
                                                    TableField('ActiveFlag', 'boolean', ['NOT NULL', 'CONSTRAINT "DF_Vendor_ActiveFlag" DEFAULT (true)'], '0 = Vendor no longer used. 1 = Vendor is actively used.'), 
                                                    TableField('PurchasingWebServiceURL', 'varchar(1024)', ['NULL'], 'Vendor URL.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Vendor_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                   TableRelation('Person', 'BusinessEntity', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES Person.BusinessEntity(BusinessEntityID)')
                                                ],
                                                [
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES Person.BusinessEntity(BusinessEntityID)',
                                                    'CONSTRAINT "CK_Vendor_CreditRating" CHECK (CreditRating BETWEEN 1 AND 5)'
                                                ],
                                                'Companies from whom Adventure Works Cycles purchases parts or other goods.'
                                            )
                                        ]
                                    ),
                                    Schema(
                                        'Sales',
                                        [
                                            Table(
                                                'Sales', 
                                                'CountryRegionCurrency', 
                                                [
                                                    TableField('CountryRegionCode', 'varchar(3)', ['NOT NULL'], 'ISO code for countries and regions. Foreign key to CountryRegion.CountryRegionCode.'),
                                                    TableField('CurrencyCode', 'char(3)', ['NOT NULL'], 'ISO standard currency code. Foreign key to Currency.CurrencyCode.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_CountryRegionCurrency_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'CountryRegion', 'CountryRegionCode', 'CountryRegionCode', 'FOREIGN KEY (CountryRegionCode) REFERENCES Person.CountryRegion(CountryRegionCode)'),
                                                    TableRelation('Sales', 'Currency', 'CurrencyCode', 'CurrencyCode', 'FOREIGN KEY (CurrencyCode) REFERENCES Sales.Currency(CurrencyCode)')
                                                ],
                                                [
                                                    'PRIMARY KEY (CountryRegionCode, CurrencyCode)',
                                                    'FOREIGN KEY (CountryRegionCode) REFERENCES Person.CountryRegion(CountryRegionCode)',
                                                    'FOREIGN KEY (CurrencyCode) REFERENCES Sales.Currency(CurrencyCode)'
                                                ],
                                                'Cross-reference table mapping ISO currency codes to a country or region.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'CreditCard', 
                                                [
                                                    TableField('CreditCardID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for CreditCard records.'), 
                                                    TableField('CardType', 'varchar(50)', ['NOT NULL'], 'Credit card name.'), 
                                                    TableField('CardNumber', 'varchar(25)', ['NOT NULL'], 'Credit card number.'), 
                                                    TableField('ExpMonth', 'smallint', ['NOT NULL'], 'Credit card expiration month.'), 
                                                    TableField('ExpYear', 'smallint', ['NOT NULL'], 'Credit card expiration year.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_CreditCard_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Customer credit card information.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'Currency', 
                                                [
                                                    TableField('CurrencyCode', 'char(3)', ['NOT NULL', 'PRIMARY KEY'], 'The ISO code for the Currency.'), 
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Currency name.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Currency_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None,
                                                None,
                                                'Lookup table containing standard ISO currencies.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'CurrencyRate', 
                                                [
                                                    TableField('CurrencyRateID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for CurrencyRate records.'), 
                                                    TableField('CurrencyRateDate', 'TIMESTAMP', ['NOT NULL'], 'Date and time the exchange rate was obtained.'), 
                                                    TableField('FromCurrencyCode', 'char(3)', ['NOT NULL'], 'Exchange rate was converted from this currency code.'), 
                                                    TableField('ToCurrencyCode', 'char(3)', ['NOT NULL'], 'Exchange rate was converted to this currency code.'), 
                                                    TableField('AverageRate', 'numeric', ['NOT NULL'], 'Average exchange rate for the day.'), 
                                                    TableField('EndOfDayRate', 'numeric', ['NOT NULL'], 'Final exchange rate for the day.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_CurrencyRate_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Sales', 'Currency', 'FromCurrencyCode', 'CurrencyCode', 'FOREIGN KEY (FromCurrencyCode) REFERENCES Sales.Currency(CurrencyCode)'),
                                                    TableRelation('Sales', 'Currency', 'ToCurrencyCode', 'CurrencyCode', 'FOREIGN KEY (ToCurrencyCode) REFERENCES Sales.Currency(CurrencyCode)')
                                                ],
                                                [
                                                    'FOREIGN KEY (FromCurrencyCode) REFERENCES Sales.Currency(CurrencyCode)',
                                                    'FOREIGN KEY (ToCurrencyCode) REFERENCES Sales.Currency(CurrencyCode)'
                                                ],
                                                'Currency exchange rates.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'Customer', 
                                                [
                                                    TableField('CustomerID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key.'), 
                                                    TableField('PersonID', 'INT', ['NULL'], 'Foreign key to Person.BusinessEntityID'), 
                                                    TableField('StoreID', 'INT', ['NULL'], 'Foreign key to Store.BusinessEntityID'), 
                                                    TableField('TerritoryID', 'INT', ['NULL'], 'ID of the territory in which the customer is located. Foreign key to SalesTerritory.SalesTerritoryID.'), 
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_Customer_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Customer_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'Person', 'PersonID', 'BusinessEntityID', 'FOREIGN KEY (PersonID) REFERENCES Person.Person(BusinessEntityID)'),
                                                    TableRelation('Sales', 'Store', 'StoreID', 'BusinessEntityID', 'FOREIGN KEY (StoreID) REFERENCES Sales.Store(BusinessEntityID)'),
                                                    TableRelation('Sales', 'SalesTerritory', 'TerritoryID', 'TerritoryID', 'FOREIGN KEY (TerritoryID) REFERENCES Sales.SalesTerritory(TerritoryID)')
                                                ],
                                                [
                                                    'FOREIGN KEY (PersonID) REFERENCES Person.Person(BusinessEntityID)',
                                                    'FOREIGN KEY (StoreID) REFERENCES Sales.Store(BusinessEntityID)',
                                                    'FOREIGN KEY (TerritoryID) REFERENCES Sales.SalesTerritory(TerritoryID)'
                                                ],
                                                'Current customer information. Also see the Person and Store tables.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'PersonCreditCard', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL'], 'Business entity identification number. Foreign key to Person.BusinessEntityID.'), 
                                                    TableField('CreditCardID', 'INT', ['NOT NULL'], 'Credit card identification number. Foreign key to CreditCard.CreditCardID.'), 
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_PersonCreditCard_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'Person', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES Person.Person(BusinessEntityID)'),
                                                    TableRelation('Sales', 'CreditCard', 'CreditCardID', 'CreditCardID', 'FOREIGN KEY (CreditCardID) REFERENCES Sales.CreditCard(CreditCardID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (BusinessEntityID, CreditCardID)',
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES Person.Person(BusinessEntityID)',
                                                    'FOREIGN KEY (CreditCardID) REFERENCES Sales.CreditCard(CreditCardID)'
                                                ],
                                                'Cross-reference table mapping people to their credit card information in the CreditCard table.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'SalesOrderDetail', 
                                                [
                                                    TableField('SalesOrderID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to SalesOrderHeader.SalesOrderID.'),
                                                    TableField('SalesOrderDetailID', 'SERIAL', ['NOT NULL'], 'Primary key. One incremental unique number per product sold.'),
                                                    TableField('CarrierTrackingNumber', 'varchar(25)', ['NULL'], 'Shipment tracking number supplied by the shipper.'),
                                                    TableField('OrderQty', 'smallint', ['NOT NULL'], 'Quantity ordered per product.'),
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Product sold to customer. Foreign key to Product.ProductID.'),
                                                    TableField('SpecialOfferID', 'INT', ['NOT NULL'], 'Promotional code. Foreign key to SpecialOffer.SpecialOfferID.'),
                                                    TableField('UnitPrice', 'numeric', ['NOT NULL'], 'Selling price of a single product.'),
                                                    TableField('UnitPriceDiscount', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SalesOrderDetail_UnitPriceDiscount" DEFAULT (0.0)'], 'Discount amount.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_SalesOrderDetail_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_SalesOrderDetail_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Sales', 'SalesOrderHeader', 'SalesOrderID', 'SalesOrderID', 'FOREIGN KEY (SalesOrderID) REFERENCES Sales.SalesOrderHeader(SalesOrderID) ON DELETE CASCADE'),
                                                    TableRelation('Sales', 'SpecialOfferProduct', ['SpecialOfferID', 'ProductID'], ['SpecialOfferID', 'ProductID'], 'FOREIGN KEY (SpecialOfferID, ProductID) REFERENCES Sales.SpecialOfferProduct(SpecialOfferID, ProductID)'),
                                                    TableRelation('Production', 'Product', 'ProductID', 'ProductID', 'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_SalesOrderDetail_OrderQty" CHECK (OrderQty > 0)',
                                                    'CONSTRAINT "CK_SalesOrderDetail_UnitPrice" CHECK (UnitPrice >= 0.00)',
                                                    'CONSTRAINT "CK_SalesOrderDetail_UnitPriceDiscount" CHECK (UnitPriceDiscount >= 0.00)',
                                                    'PRIMARY KEY (SalesOrderID, SalesOrderDetailID)',
                                                    'FOREIGN KEY (SalesOrderID) REFERENCES Sales.SalesOrderHeader(SalesOrderID) ON DELETE CASCADE',
                                                    'FOREIGN KEY (SpecialOfferID, ProductID) REFERENCES Sales.SpecialOfferProduct(SpecialOfferID, ProductID)',
                                                    'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)'
                                                ],
                                                'Individual products associated with a specific sales order. See SalesOrderHeader.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'SalesOrderHeader', 
                                                [
                                                    TableField('SalesOrderID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key.'),
                                                    TableField('RevisionNumber', 'smallint', ['NOT NULL', 'CONSTRAINT "DF_SalesOrderHeader_RevisionNumber" DEFAULT (0)'], 'Incremental number to track changes to the sales order over time.'),
                                                    TableField('OrderDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_SalesOrderHeader_OrderDate" DEFAULT (NOW())'], 'Dates the sales order was created.'),
                                                    TableField('DueDate', 'TIMESTAMP', ['NOT NULL'], 'Date the order is due to the customer.'),
                                                    TableField('ShipDate', 'TIMESTAMP', ['NULL'], 'Date the order was shipped to the customer.'),
                                                    TableField('Status', 'smallint', ['NOT NULL', 'CONSTRAINT "DF_SalesOrderHeader_Status" DEFAULT (1)'], 'Order current status. 1 = In process; 2 = Approved; 3 = Backordered; 4 = Rejected; 5 = Shipped; 6 = Cancelled'),
                                                    TableField('OnlineOrderFlag', 'boolean', ['NOT NULL', 'CONSTRAINT "DF_SalesOrderHeader_OnlineOrderFlag" DEFAULT (true)'], '0 = Order placed by sales person. 1 = Order placed online by customer.'),
                                                    TableField('PurchaseOrderNumber', 'varchar(25)', ['NULL'], 'Customer purchase order number reference.'),
                                                    TableField('AccountNumber', 'varchar(15)', ['NULL'], 'Financial accounting number reference.'),
                                                    TableField('CustomerID', 'INT', ['NOT NULL'], 'Customer identification number. Foreign key to Customer.BusinessEntityID.'),
                                                    TableField('SalesPersonID', 'INT', ['NULL'], 'Sales person who created the sales order. Foreign key to SalesPerson.BusinessEntityID.'),
                                                    TableField('TerritoryID', 'INT', ['NULL'], 'Territory in which the sale was made. Foreign key to SalesTerritory.SalesTerritoryID.'),
                                                    TableField('BillToAddressID', 'INT', ['NOT NULL'], 'Customer billing address. Foreign key to Address.AddressID.'),
                                                    TableField('ShipToAddressID', 'INT', ['NOT NULL'], 'Customer shipping address. Foreign key to Address.AddressID.'),
                                                    TableField('ShipMethodID', 'INT', ['NOT NULL'], 'Shipping method. Foreign key to ShipMethod.ShipMethodID.'),
                                                    TableField('CreditCardID', 'INT', ['NULL'], 'Credit card identification number. Foreign key to CreditCard.CreditCardID.'),
                                                    TableField('CreditCardApprovalCode', 'varchar(15)', ['NULL'], 'Approval code provided by the credit card company.'),
                                                    TableField('CurrencyRateID', 'INT', ['NULL'], 'Currency exchange rate used. Foreign key to CurrencyRate.CurrencyRateID.'),
                                                    TableField('SubTotal', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SalesOrderHeader_SubTotal" DEFAULT (0.00)'], 'Sales subtotal. Computed as SUM(SalesOrderDetail.LineTotal) for the appropriate SalesOrderID.'),
                                                    TableField('TaxAmt', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SalesOrderHeader_TaxAmt" DEFAULT (0.00)'], 'Tax amount.'),
                                                    TableField('Freight', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SalesOrderHeader_Freight" DEFAULT (0.00)'], 'Shipping cost.'),
                                                    TableField('TotalDue', 'numeric', None, 'Total due from customer. Computed as Subtotal + TaxAmt + Freight.'),
                                                    TableField('Comment', 'varchar(128)', ['NULL'], 'Sales representative comments.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_SalesOrderHeader_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_SalesOrderHeader_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'Address', 'BillToAddressID', 'AddressID', 'FOREIGN KEY (BillToAddressID) REFERENCES Person.Address(AddressID)'),
                                                    TableRelation('Person', 'Address', 'ShipToAddressID', 'AddressID', 'FOREIGN KEY (ShipToAddressID) REFERENCES Person.Address(AddressID)'),
                                                    TableRelation('Sales', 'CreditCard', 'CreditCardID', 'CreditCardID', 'FOREIGN KEY (CreditCardID) REFERENCES Sales.CreditCard(CreditCardID)'),
                                                    TableRelation('Sales', 'CurrencyRate', 'CurrencyRateID', 'CurrencyRateID', 'FOREIGN KEY (CurrencyRateID) REFERENCES Sales.CurrencyRate(CurrencyRateID)'),
                                                    TableRelation('Sales', 'Customer', 'CustomerID', 'CustomerID', 'FOREIGN KEY (CustomerID) REFERENCES Sales.Customer(CustomerID)'),
                                                    TableRelation('Sales', 'SalesPerson', 'SalesPersonID', 'BusinessEntityID', 'FOREIGN KEY (SalesPersonID) REFERENCES Sales.SalesPerson(BusinessEntityID)'),
                                                    TableRelation('Purchasing', 'ShipMethod', 'ShipMethodID', 'ShipMethodID', 'FOREIGN KEY (ShipMethodID) REFERENCES Purchasing.ShipMethod(ShipMethodID)'),
                                                    TableRelation('Sales', 'SalesTerritory', 'TerritoryID', 'TerritoryID', 'FOREIGN KEY (TerritoryID) REFERENCES Sales.SalesTerritory(TerritoryID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_SalesOrderHeader_Status" CHECK (Status BETWEEN 0 AND 8)',
                                                    'CONSTRAINT "CK_SalesOrderHeader_DueDate" CHECK (DueDate >= OrderDate)',
                                                    'CONSTRAINT "CK_SalesOrderHeader_ShipDate" CHECK ((ShipDate >= OrderDate) OR (ShipDate IS NULL))',
                                                    'CONSTRAINT "CK_SalesOrderHeader_SubTotal" CHECK (SubTotal >= 0.00)',
                                                    'CONSTRAINT "CK_SalesOrderHeader_TaxAmt" CHECK (TaxAmt >= 0.00)',
                                                    'CONSTRAINT "CK_SalesOrderHeader_Freight" CHECK (Freight >= 0.00)',
                                                    'FOREIGN KEY (BillToAddressID) REFERENCES Person.Address(AddressID)',
                                                    'FOREIGN KEY (ShipToAddressID) REFERENCES Person.Address(AddressID)',
                                                    'FOREIGN KEY (CreditCardID) REFERENCES Sales.CreditCard(CreditCardID)',
                                                    'FOREIGN KEY (CurrencyRateID) REFERENCES Sales.CurrencyRate(CurrencyRateID)',
                                                    'FOREIGN KEY (CustomerID) REFERENCES Sales.Customer(CustomerID)',
                                                    'FOREIGN KEY (SalesPersonID) REFERENCES Sales.SalesPerson(BusinessEntityID)',
                                                    'FOREIGN KEY (ShipMethodID) REFERENCES Purchasing.ShipMethod(ShipMethodID)',
                                                    'FOREIGN KEY (TerritoryID) REFERENCES Sales.SalesTerritory(TerritoryID)'
                                                ],
                                                'General sales order information.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'SalesOrderHeaderSalesReason', 
                                                [
                                                    TableField('SalesOrderID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to SalesOrderHeader.SalesOrderID.'),
                                                    TableField('SalesReasonID', 'INT', ['NOT NULL'], 'Primary key. Foreign key to SalesReason.SalesReasonID.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_SalesOrderHeaderSalesReason_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Sales', 'SalesReason', 'SalesReasonID', 'SalesReasonID', 'FOREIGN KEY (SalesReasonID) REFERENCES Sales.SalesReason(SalesReasonID)'),
                                                    TableRelation('Sales', 'SalesOrderHeader', 'SalesOrderID', 'SalesOrderID', 'FOREIGN KEY (SalesOrderID) REFERENCES Sales.SalesOrderHeader(SalesOrderID) ON DELETE CASCADE')
                                                ],
                                                [
                                                    'PRIMARY KEY (SalesOrderID, SalesReasonID)',
                                                    'FOREIGN KEY (SalesReasonID) REFERENCES Sales.SalesReason(SalesReasonID)',
                                                    'FOREIGN KEY (SalesOrderID) REFERENCES Sales.SalesOrderHeader(SalesOrderID) ON DELETE CASCADE'
                                                ],
                                                'Cross-reference table mapping sales orders to sales reason codes.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'SalesPerson', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for SalesPerson records. Foreign key to Employee.BusinessEntityID'),
                                                    TableField('TerritoryID', 'INT', ['NULL'], 'Territory currently assigned to. Foreign key to SalesTerritory.SalesTerritoryID.'),
                                                    TableField('SalesQuota', 'numeric', ['NULL'], 'Projected yearly sales.'),
                                                    TableField('Bonus', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SalesPerson_Bonus" DEFAULT (0.00)'], 'Bonus due if quota is met.'),
                                                    TableField('CommissionPct', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SalesPerson_CommissionPct" DEFAULT (0.00)'], 'Commission percent received per sale.'),
                                                    TableField('SalesYTD', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SalesPerson_SalesYTD" DEFAULT (0.00)'], 'Sales total year to date.'),
                                                    TableField('SalesLastYear', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SalesPerson_SalesLastYear" DEFAULT (0.00)'], 'Sales total of previous year.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_SalesPerson_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_SalesPerson_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('HumanResources', 'Employee', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES HumanResources.Employee(BusinessEntityID)'),
                                                    TableRelation('Sales', 'SalesTerritory', 'TerritoryID', 'TerritoryID', 'FOREIGN KEY (TerritoryID) REFERENCES Sales.SalesTerritory(TerritoryID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_SalesPerson_SalesQuota" CHECK (SalesQuota > 0.00)',
                                                    'CONSTRAINT "CK_SalesPerson_Bonus" CHECK (Bonus >= 0.00)',
                                                    'CONSTRAINT "CK_SalesPerson_CommissionPct" CHECK (CommissionPct >= 0.00)',
                                                    'CONSTRAINT "CK_SalesPerson_SalesYTD" CHECK (SalesYTD >= 0.00)',
                                                    'CONSTRAINT "CK_SalesPerson_SalesLastYear" CHECK (SalesLastYear >= 0.00)',
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES HumanResources.Employee(BusinessEntityID)',
                                                    'FOREIGN KEY (TerritoryID) REFERENCES Sales.SalesTerritory(TerritoryID)'
                                                ],
                                                'Sales representative current information.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'SalesPersonQuotaHistory', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL'], 'Sales person identification number. Foreign key to SalesPerson.BusinessEntityID.'),
                                                    TableField('QuotaDate', 'TIMESTAMP', ['NOT NULL'], 'Sales quota date.'),
                                                    TableField('SalesQuota', 'numeric', ['NOT NULL'], 'Sales quota amount.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_SalesPersonQuotaHistory_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_SalesPersonQuotaHistory_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Sales', 'SalesPerson', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES Sales.SalesPerson(BusinessEntityID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_SalesPersonQuotaHistory_SalesQuota" CHECK (SalesQuota > 0.00)',
                                                    'PRIMARY KEY (BusinessEntityID, QuotaDate)',
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES Sales.SalesPerson(BusinessEntityID)'
                                                ],
                                                'Sales performance tracking.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'SalesReason', 
                                                [
                                                    TableField('SalesReasonID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for SalesReason records.'),
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Sales reason description.'),
                                                    TableField('ReasonType', 'varchar(50)', ['NOT NULL'], 'Category the sales reason belongs to.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_SalesReason_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None, 
                                                None,  
                                                'Lookup table of customer purchase reasons.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'SalesTaxRate', 
                                                [
                                                    TableField('SalesTaxRateID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for SalesTaxRate records.'),
                                                    TableField('StateProvinceID', 'INT', ['NOT NULL'], 'State, province, or country/region the sales tax applies to.'),
                                                    TableField('TaxType', 'smallint', ['NOT NULL'], '1 = Tax applied to retail transactions, 2 = Tax applied to wholesale transactions, 3 = Tax applied to all sales (retail and wholesale) transactions.'),
                                                    TableField('TaxRate', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SalesTaxRate_TaxRate" DEFAULT (0.00)'], 'Tax rate amount.'),
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Tax rate description.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_SalesTaxRate_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_SalesTaxRate_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'StateProvince', 'StateProvinceID', 'StateProvinceID', 'FOREIGN KEY (StateProvinceID) REFERENCES Person.StateProvince(StateProvinceID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_SalesTaxRate_TaxType" CHECK (TaxType BETWEEN 1 AND 3)',
                                                    'FOREIGN KEY (StateProvinceID) REFERENCES Person.StateProvince(StateProvinceID)'
                                                ],
                                                'Tax rate lookup table.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'SalesTerritory', 
                                                [
                                                    TableField('TerritoryID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for SalesTerritory records.'),
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Sales territory description'),
                                                    TableField('CountryRegionCode', 'varchar(3)', ['NOT NULL'], 'ISO standard country or region code. Foreign key to CountryRegion.CountryRegionCode.'),
                                                    TableField('"group"', 'varchar(50)', ['NOT NULL'], 'Geographic area to which the sales territory belong.'),
                                                    TableField('SalesYTD', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SalesTerritory_SalesYTD" DEFAULT (0.00)'], 'Sales in the territory year to date.'),
                                                    TableField('SalesLastYear', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SalesTerritory_SalesLastYear" DEFAULT (0.00)'], 'Sales in the territory the previous year.'),
                                                    TableField('CostYTD', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SalesTerritory_CostYTD" DEFAULT (0.00)'], 'Business costs in the territory year to date.'),
                                                    TableField('CostLastYear', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SalesTerritory_CostLastYear" DEFAULT (0.00)'], 'Business costs in the territory the previous year.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_SalesTerritory_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_SalesTerritory_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'CountryRegion', 'CountryRegionCode', 'CountryRegionCode', 'FOREIGN KEY (CountryRegionCode) REFERENCES Person.CountryRegion(CountryRegionCode)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_SalesTerritory_SalesYTD" CHECK (SalesYTD >= 0.00)',
                                                    'CONSTRAINT "CK_SalesTerritory_SalesLastYear" CHECK (SalesLastYear >= 0.00)',
                                                    'CONSTRAINT "CK_SalesTerritory_CostYTD" CHECK (CostYTD >= 0.00)',
                                                    'CONSTRAINT "CK_SalesTerritory_CostLastYear" CHECK (CostLastYear >= 0.00)',
                                                    'FOREIGN KEY (CountryRegionCode) REFERENCES Person.CountryRegion(CountryRegionCode)'
                                                ],
                                                'Sales territory lookup table.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'SalesTerritoryHistory', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL'], 'Primary key. The sales rep. Foreign key to SalesPerson.BusinessEntityID.'),
                                                    TableField('TerritoryID', 'INT', ['NOT NULL'], 'Primary key. Territory identification number. Foreign key to SalesTerritory.SalesTerritoryID.'),
                                                    TableField('StartDate', 'TIMESTAMP', ['NOT NULL'], 'Primary key. Date the sales representative started work in the territory.'),
                                                    TableField('EndDate', 'TIMESTAMP', ['NULL'], 'Date the sales representative left work in the territory.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_SalesTerritoryHistory_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_SalesTerritoryHistory_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Sales', 'SalesPerson', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES Sales.SalesPerson(BusinessEntityID)'),
                                                    TableRelation('Sales', 'SalesTerritory', 'TerritoryID', 'TerritoryID', 'FOREIGN KEY (TerritoryID) REFERENCES Sales.SalesTerritory(TerritoryID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_SalesTerritoryHistory_EndDate" CHECK ((EndDate >= StartDate) OR (EndDate IS NULL))',
                                                    'PRIMARY KEY (BusinessEntityID, StartDate, TerritoryID)',
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES Sales.SalesPerson(BusinessEntityID)',
                                                    'FOREIGN KEY (TerritoryID) REFERENCES Sales.SalesTerritory(TerritoryID)'
                                                ],
                                                'Sales representative transfers to other sales territories.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'ShoppingCartItem', 
                                                [
                                                    TableField('ShoppingCartItemID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for ShoppingCartItem records.'),
                                                    TableField('ShoppingCartID', 'varchar(50)', ['NOT NULL'], 'Shopping cart identification number.'),
                                                    TableField('Quantity', 'INT', ['NOT NULL', 'CONSTRAINT "DF_ShoppingCartItem_Quantity" DEFAULT (1)'], 'Product quantity ordered.'),
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Product ordered. Foreign key to Product.ProductID.'),
                                                    TableField('DateCreated', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ShoppingCartItem_DateCreated" DEFAULT (NOW())'], 'Date the time the record was created.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_ShoppingCartItem_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'Product', 'ProductID', 'ProductID', 'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)')
                                                ],
                                                [
                                                    'CONSTRAINT "CK_ShoppingCartItem_Quantity" CHECK (Quantity >= 1)',
                                                    'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)'
                                                ],
                                                'Contains online customer orders until the order is submitted or cancelled.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'SpecialOffer', 
                                                [
                                                    TableField('SpecialOfferID', 'SERIAL', ['NOT NULL', 'PRIMARY KEY'], 'Primary key for SpecialOffer records.'),
                                                    TableField('Description', 'varchar(255)', ['NOT NULL'], 'Discount description.'),
                                                    TableField('DiscountPct', 'numeric', ['NOT NULL', 'CONSTRAINT "DF_SpecialOffer_DiscountPct" DEFAULT (0.00)'], 'Discount percentage.'),
                                                    TableField('Type', 'varchar(50)', ['NOT NULL'], 'Discount type category.'),
                                                    TableField('Category', 'varchar(50)', ['NOT NULL'], 'Group the discount applies to such as Reseller or Customer.'),
                                                    TableField('StartDate', 'TIMESTAMP', ['NOT NULL'], 'Discount start date.'),
                                                    TableField('EndDate', 'TIMESTAMP', ['NOT NULL'], 'Discount end date.'),
                                                    TableField('MinQty', 'INT', ['NOT NULL', 'CONSTRAINT "DF_SpecialOffer_MinQty" DEFAULT (0)'], 'Minimum discount percent allowed.'),
                                                    TableField('MaxQty', 'INT', ['NULL'], 'Maximum discount percent allowed.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_SpecialOffer_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_SpecialOffer_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                None, 
                                                [
                                                    'CONSTRAINT "CK_SpecialOffer_EndDate" CHECK (EndDate >= StartDate)',
                                                    'CONSTRAINT "CK_SpecialOffer_DiscountPct" CHECK (DiscountPct >= 0.00)',
                                                    'CONSTRAINT "CK_SpecialOffer_MinQty" CHECK (MinQty >= 0)',
                                                    'CONSTRAINT "CK_SpecialOffer_MaxQty" CHECK (MaxQty >= 0 OR MaxQty IS NULL)'
                                                ],
                                                'Sale discounts lookup table.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'SpecialOfferProduct', 
                                                [
                                                    TableField('SpecialOfferID', 'INT', ['NOT NULL'], 'Primary key for SpecialOfferProduct records.'),
                                                    TableField('ProductID', 'INT', ['NOT NULL'], 'Product identification number. Foreign key to Product.ProductID.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_SpecialOfferProduct_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_SpecialOfferProduct_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Production', 'Product', 'ProductID', 'ProductID', 'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)'),
                                                    TableRelation('Sales', 'SpecialOffer', 'SpecialOfferID', 'SpecialOfferID', 'FOREIGN KEY (SpecialOfferID) REFERENCES Sales.SpecialOffer(SpecialOfferID)')
                                                ],
                                                [
                                                    'PRIMARY KEY (SpecialOfferID, ProductID)',
                                                    'FOREIGN KEY (ProductID) REFERENCES Production.Product(ProductID)',
                                                    'FOREIGN KEY (SpecialOfferID) REFERENCES Sales.SpecialOffer(SpecialOfferID)'
                                                ],
                                                'Cross-reference table mapping products to special offer discounts.'
                                            ),
                                            Table(
                                                'Sales', 
                                                'Store', 
                                                [
                                                    TableField('BusinessEntityID', 'INT', ['NOT NULL', 'PRIMARY KEY'], 'Primary key. Foreign key to Customer.BusinessEntityID.'),
                                                    TableField('Name', 'varchar(50)', ['NOT NULL'], 'Name of the store.'),
                                                    TableField('SalesPersonID', 'INT', ['NULL'], 'ID of the sales person assigned to the customer. Foreign key to SalesPerson.BusinessEntityID.'),
                                                    TableField('Demographics', 'XML', ['NULL'], 'Demographic informationg about the store such as the number of employees, annual sales and store type.'),
                                                    TableField('rowguid', 'uuid', ['NOT NULL', 'CONSTRAINT "DF_Store_rowguid" DEFAULT (uuid_generate_v1())'], 'ROWGUIDCOL number uniquely identifying the record. Required for FileStream.'),
                                                    TableField('ModifiedDate', 'TIMESTAMP', ['NOT NULL', 'CONSTRAINT "DF_Store_ModifiedDate" DEFAULT (NOW())'], None)
                                                ],
                                                [
                                                    TableRelation('Person', 'BusinessEntity', 'BusinessEntityID', 'BusinessEntityID', 'FOREIGN KEY (BusinessEntityID) REFERENCES Person.BusinessEntity(BusinessEntityID)'),
                                                    TableRelation('Sales', 'SalesPerson', 'SalesPersonID', 'BusinessEntityID', 'FOREIGN KEY (SalesPersonID) REFERENCES Sales.SalesPerson(BusinessEntityID)')
                                                ],
                                                [
                                                    'FOREIGN KEY (BusinessEntityID) REFERENCES Person.BusinessEntity(BusinessEntityID)',
                                                    'FOREIGN KEY (SalesPersonID) REFERENCES Sales.SalesPerson(BusinessEntityID)'
                                                ],
                                                'Customers (resellers) of Adventure Works products.'
                                            )
                                        ]
                                    )
                                ]
                        )
        return database

    @staticmethod
    def extract_tables_from_data(s):
        """
        Extracts tables from data
        """
        # Pattern explanation:
        # \. matches a literal dot.
        # \S+ matches one or more non-whitespace characters.
        # Lookahead for an opening parenthesis without including it in the match.
        pattern = r'\.(\S+?)(?=\()'
        matches = re.findall(pattern, s.replace(" ", ""))
        return matches

    @staticmethod
    def calculate_table_embeddings(database, model, type = 'TABLE_DEFINITION'):
        if type == 'TABLE_DEFINITION':
            for schema in database.schemata:
                for table in schema.tables:
                    table.embedding = model.encode(table.get_definition(ignore_descriptions = True,
                                                                                 single_line = True,
                                                                                 ignore_table_constraints = False,
                                                                                 ignore_field_constraints = False,
                                                                                 ignore_primary_key_constraints = False,
                                                                                 ignore_foreign_key_constraints = False,
                                                                                 fields_to_ignore = ['rowguid', 'ModifiedDate']
                                                                                 ))
                    
        elif type == 'TABLE_DEFINITION_IGNORE_CONSTRAINTS':
            for schema in database.schemata:
                for table in schema.tables:
                    table.embedding = model.encode(table.get_definition(ignore_descriptions = True,
                                                                                 single_line = True,
                                                                                 ignore_table_constraints = True,
                                                                                 ignore_field_constraints = True,
                                                                                 ignore_primary_key_constraints = True,
                                                                                 ignore_foreign_key_constraints = False,
                                                                                 fields_to_ignore = ['rowguid', 'ModifiedDate']
                                                                                 ))
                    
        elif type == 'TABLE_DESCRIPTION':
            for schema in database.schemata:
                for table in schema.tables:
                    table.embedding = model.encode(table.description)

        elif type == 'TABLE_COLUMN_DESCRIPTION':
            for schema in database.schemata:
                for table in schema.tables:
                    table.embedding = model.encode(table.get_description_including_columns())
                    
        elif type == 'TABLE_DEFINITION_DESCRIPTIONS':
            for schema in database.schemata:
                for table in schema.tables:
                    table.embedding = model.encode(table.get_definition(ignore_descriptions = False,
                                                                                 single_line = False,
                                                                                 ignore_table_constraints = True,
                                                                                 ignore_field_constraints = True,
                                                                                 ignore_primary_key_constraints = True,
                                                                                 ignore_foreign_key_constraints = True,
                                                                                 fields_to_ignore = ['rowguid', 'ModifiedDate']
                                                                                 ))

        else: raise Exception("Invalid embedding calculation type")
        
    @staticmethod
    def calculate_tables_question_cosine_similarity(database, question, model):
        question_embedding = model.encode(question)

        #Calculate table similarity with question
        for schema in database.schemata:
            for table in schema.tables:
                table.cosine_similarity = 1 - distance.cosine(table.embedding, question_embedding)
        
    @staticmethod
    def calculate_table_scores(database, type = 'SIMPLE'):
        for schema in database.schemata:
                for table in schema.tables:
                    table.score = table.cosine_similarity
                    
        if type == 'SIMPLE':
            return
        
        if type == 'ECHO':
            for schema in database.schemata:
                for table in schema.tables:
                    table.neighbor_score = sum([table.cosine_similarity * neighbor.cosine_similarity for neighbor in database.get_referenced_tables(table) + database.get_tables_referencing_table(table)])
                    table.score = table.cosine_similarity + table.neighbor_score
                    
        if type == 'ECHO_V2':
            top_10_tables = sorted([table for schema in database.schemata for table in schema.tables], key=lambda table: table.cosine_similarity, reverse=True)[0:10]
            
            for schema in database.schemata:
                for table in schema.tables:
                    table.neighbor_score = sum([table.cosine_similarity * neighbor.cosine_similarity for neighbor in database.get_referenced_tables(table) + database.get_tables_referencing_table(table) if neighbor in top_10_tables])
                    table.score = table.cosine_similarity + table.neighbor_score
                
        if type == 'ECHO_V3':
            for schema in database.schemata:
                for table in schema.tables:
                    neighbors_1 = set([neighbor for neighbor in database.get_referenced_tables(table) + database.get_tables_referencing_table(table)])
                    neighbors_2 = set([neighbor_2 for neighbor_1 in neighbors_1 for neighbor_2 in database.get_referenced_tables(neighbor_1) + database.get_tables_referencing_table(neighbor_1) if neighbor_2 not in neighbors_1 and neighbor_2 != table])
                    
                    for neighbor in neighbors_1:
                        neighbor.score += table.cosine_similarity * neighbor.cosine_similarity
                    
                    for neighbor in neighbors_2:
                        neighbor.score += 0.5 * table.cosine_similarity * neighbor.cosine_similarity
                        
        if type == 'ECHO_V4':
            top_10_tables = sorted([table for schema in database.schemata for table in schema.tables], key=lambda table: table.cosine_similarity, reverse=True)[0:10]
            
            for table in top_10_tables:
                neighbors_1 = set([neighbor for neighbor in database.get_referenced_tables(table) + database.get_tables_referencing_table(table)])
                neighbors_2 = set([neighbor_2 for neighbor_1 in neighbors_1 for neighbor_2 in database.get_referenced_tables(neighbor_1) + database.get_tables_referencing_table(neighbor_1) if neighbor_2 not in neighbors_1 and neighbor_2 != table])
                
                for neighbor in neighbors_1:
                    neighbor.score += table.cosine_similarity * neighbor.cosine_similarity
                
                for neighbor in neighbors_2:
                    neighbor.score += 0.5 * table.cosine_similarity * neighbor.cosine_similarity      
                
    
    @staticmethod
    def initialize_beams(beam_width, database, dynamic_threshold, initializer = 'COSINE_SIMILARITY'):
        
        if initializer == 'COSINE_SIMILARITY':
            tables_sorted = sorted([table for schema in database.schemata for table in schema.tables], key=lambda table: table.cosine_similarity, reverse=True)
            
        elif initializer == 'SCORE':
            tables_sorted = sorted([table for schema in database.schemata for table in schema.tables], key=lambda table: table.score, reverse=True)
            
        else:
            raise Exception('Initializer must be COSINE_SIMILARITY or SCORE')

        beams = []
        if beam_width:
            for i in range(0, beam_width):
                beam = Beam()
                beam.tables.append(tables_sorted[i])
                beam.score = tables_sorted[i].score
                beams.append(beam)

        if dynamic_threshold:
            qualifying_tables = []
            
            if initializer == 'COSINE_SIMILARITY':
                highest_cosine_similarity = tables_sorted[0].cosine_similarity
                qualifying_tables = [table for schema in database.schemata for table in schema.tables if table.cosine_similarity >= highest_cosine_similarity * dynamic_threshold]
            if initializer == 'SCORE':
                highest_score = tables_sorted[0].score
                qualifying_tables = [table for schema in database.schemata for table in schema.tables if table.score >= highest_score * dynamic_threshold]
            
            for table in qualifying_tables:
                beam = Beam()
                beam.tables.append(table)
                beam.score = table.score
                beams.append(beam)

        for beam in beams:
          beam.cosine_similarity = beam.tables[0].cosine_similarity
        
        return beams
        
    
    @staticmethod
    def calculate_beams(beams, beam_width, beam_length, dynamic_threshold, database, type = 'AC_EMBEDDING_SIMILARITY', verbose = False):
        if beam_length > 1:
            if type == 'AC_EMBEDDING_SIMILARITY':
                for i in range(0, beam_length - 1):
                    possible_beams = []
                    for beam in beams:

                      ## print
                      if verbose:
                        print("Beam: ", end = '')
                        for table in beam.tables:
                          print(table.get_scores_for_dbeam() + ' - ', end = '')
                        print(str(beam.score))
                      ## ~~~

                      possible_tables = []
                      for beamTable in beam.tables:

                        possible_tables = set(database.get_referenced_tables(beamTable) + database.get_tables_referencing_table(beamTable))

                        for table in beam.tables:
                            possible_tables.discard(table)

                      if len(possible_tables) == 0:
                        ## If no tables (that are not already in beam) are present
                        if verbose:
                            print("Connectivity Break")
                        possibleBeam = Beam()

                        ## Then choose the top-beam_length tables from the database with the highest similarity
                        possible_tables_sorted = sorted([table for schema in database.schemata for table in schema.tables if table not in set(beam.tables)], key=lambda table: table.score, reverse=True)

                        possible_tables = possible_tables_sorted[:beam_length]

                      for possibleTable in possible_tables:
                        possibleBeam = Beam()
                        possibleBeam.tables = beam.tables + [possibleTable]
                        if set(possibleBeam.tables) not in [set(beam.tables) for beam in possible_beams]:

                          ## print
                          if verbose:
                            print("Possible Beam: ", end = '')
                            for table1 in possibleBeam.tables:
                              print(table1.get_scores_for_dbeam() + ' - ', end = '')
                          ## ~~~

                          possibleBeam.score = beam.score + possibleTable.score
                          if verbose:
                            print(str(possibleBeam.score))
                          possible_beams.append(possibleBeam)

                      if verbose:
                        print('~~~~~~~~')

                    possible_beams.sort(key=lambda beam: beam.score, reverse = True)                
                    beams = possible_beams[0:beam_width]

                    # Print
                    if verbose:
                        print('##########')
                        print("Selected Beams: ")
                        print('##########')
                        for beam in beams:
                          for table in beam.tables:
                            print(table.get_scores_for_dbeam() + " - ", end = "")
                          print(str(beam.score))
                        print('##########')
                        print()

            #Added connectivity break when beam cannot be extended
            if type == 'AC_EMBEDDING_SIMILARITY_V2':
                for i in range(0, beam_length - 1):
                    possible_beams = []
                    for beam in beams:

                      ## print
                      if verbose:
                        print("Beam: ", end = '')
                        for table in beam.tables:
                          print(table.get_scores_for_dbeam() + ' - ', end = '')
                        print(str(beam.score))
                      ## ~~~

                      possible_tables = []
                      for beamTable in beam.tables:

                        possible_tables = set(database.get_referenced_tables(beamTable) + database.get_tables_referencing_table(beamTable))

                        for table in beam.tables:
                            possible_tables.discard(table)

                      if len(possible_tables) == 0:
                        ## If no tables (that are not already in beam) are present
                        if verbose:
                            print("Connectivity Break due to lack of connected tables")

                        ## Then choose the top-beam_length tables from the database with the highest similarity
                        possible_tables_sorted = sorted([table for schema in database.schemata for table in schema.tables if table not in set(beam.tables)], key=lambda table: table.score, reverse=True)

                        possible_tables = possible_tables_sorted[:beam_length]

                      for possibleTable in possible_tables:
                        possibleBeam = Beam()
                        possibleBeam.tables = beam.tables + [possibleTable]
                        if set(possibleBeam.tables) not in [set(beam.tables) for beam in possible_beams]:

                          ## print
                          if verbose:
                            print("Possible Beam: ", end = '')
                            for table1 in possibleBeam.tables:
                              print(table1.get_scores_for_dbeam() + ' - ', end = '')
                          ## ~~~

                          possibleBeam.score = beam.score + possibleTable.score
                          if verbose:
                            print(str(possibleBeam.score))
                          possible_beams.append(possibleBeam)

                      if verbose:
                        print('~~~~~~~~')

                    possible_beams.sort(key=lambda beam: beam.score, reverse = True)                
                    beams = possible_beams[0:beam_width]           
                    # Print
                    if verbose:
                        print('##########')
                        print("Selected Beams: ")
                        print('##########')
                        for beam in beams:
                          for table in beam.tables:
                            print(table.get_scores_for_dbeam() + " - ", end = "")
                          print(str(beam.score))
                        print('##########')
                        print()

            #V2, but beams extend from any table, instead of just last (V2)
            if type == 'AC_EMBEDDING_SIMILARITY_V3':
                for i in range(0, beam_length - 1):
                    possible_beams = []
                    for beam in beams:

                      ## print
                      if verbose:
                        print("Beam: ", end = '')
                        for table in beam.tables:
                          print(table.get_scores_for_dbeam() + ' - ', end = '')
                        print(str(beam.score))
                      ## ~~~

                      possible_tables = []
                      for beamTable in beam.tables:

                        possible_tables += database.get_referenced_tables(beamTable) + database.get_tables_referencing_table(beamTable)

                      possible_tables = set(possible_tables)

                      for table in beam.tables:
                          possible_tables.discard(table)

                      if len(possible_tables) == 0:
                        ## If no tables (that are not already in beam) are present 
                        if verbose:
                            print("Connectivity Break due to lack of connected tables")

                        ## Then choose the top-beam_length tables from the database with the highest similarity
                        possible_tables_sorted = sorted([table for schema in database.schemata for table in schema.tables if table not in set(beam.tables)], key=lambda table: table.score, reverse=True)

                        possible_tables = possible_tables_sorted[:beam_length]

                      for possibleTable in possible_tables:
                        possibleBeam = Beam()
                        possibleBeam.tables = beam.tables + [possibleTable]
                        if set(possibleBeam.tables) not in [set(beam.tables) for beam in possible_beams]:

                          ## print
                          if verbose:
                            print("Possible Beam: ", end = '')
                            for table1 in possibleBeam.tables:
                              print(table1.get_scores_for_dbeam() + ' - ', end = '')
                          ## ~~~

                          possibleBeam.score = beam.score + possibleTable.score

                          if verbose:
                            print(str(possibleBeam.score))

                          possible_beams.append(possibleBeam)
                      if verbose:
                        print('~~~~~~~~')

                    possible_beams.sort(key=lambda beam: beam.score, reverse = True)                
                    beams = possible_beams[0:beam_width]

                    # Print
                    if verbose:
                        print('##########')
                        print("Selected Beams: ")
                        print('##########')
                        for beam in beams:
                          for table in beam.tables:
                            print(table.get_scores_for_dbeam() + " - ", end = "")
                          print(str(beam.score))
                        print('##########')
                        print()
        
            #V3, but with dynamic beam width
            if type == 'AC_EMBEDDING_SIMILARITY_V4':
                for i in range(0, beam_length - 1):
                    possible_beams = []
                    for beam in beams:

                      ## print
                      if verbose:
                        print("Beam: ", end = '')
                        for table in beam.tables:
                          print(table.get_scores_for_dbeam() + ' - ', end = '')
                        print(str(beam.score))
                      ## ~~~

                      possible_tables = []
                      for beamTable in beam.tables:

                        possible_tables += database.get_referenced_tables(beamTable) + database.get_tables_referencing_table(beamTable)

                      possible_tables = set(possible_tables)

                      for table in beam.tables:
                          possible_tables.discard(table)

                      if len(possible_tables) == 0:
                        ## If no tables (that are not already in beam) are present
                        if verbose:
                            print("Connectivity Break due to lack of connected tables")

                        ## Then choose the top-beam_length tables from the database with the highest similarity
                        possible_tables_sorted = sorted([table for schema in database.schemata for table in schema.tables if table not in set(beam.tables)], key=lambda table: table.score, reverse=True)

                        possible_tables = possible_tables_sorted[:beam_length]

                      for possibleTable in possible_tables:
                        possibleBeam = Beam()
                        possibleBeam.tables = beam.tables + [possibleTable]
                        if set(possibleBeam.tables) not in [set(beam.tables) for beam in possible_beams]:

                          ## print
                          if verbose:
                            print("Possible Beam: ", end = '')
                            for table1 in possibleBeam.tables:
                              print(table1.get_scores_for_dbeam() + ' - ', end = '')
                          ## ~~~

                          possibleBeam.score = beam.score + possibleTable.score
                          if verbose:
                            print(str(possibleBeam.score))
                          possible_beams.append(possibleBeam)

                      if verbose:
                        print('~~~~~~~~')

                    #Sort by last table score
                    possible_beams.sort(key=lambda beam: beam.tables[-1].score, reverse = True)  
                    #highest score
                    highest_score = possible_beams[0].tables[-1].score
                    #Keep beams with last table over threshold
                    beams = [beam for beam in possible_beams if beam.tables[-1].score >= highest_score * dynamic_threshold]
                    beams.sort(key=lambda beam: beam.score, reverse = True)

                    # Print
                    if verbose:
                        print('##########')
                        print("Selected Beams: ")
                        print('##########')
                        for beam in beams:
                          for table in beam.tables:
                            print(table.get_scores_for_dbeam() + " - ", end = "")
                          print(str(beam.score))
                        print('##########')
                        print()
        
        
        beams.sort(key=lambda beam: beam.score, reverse = True)
        
        return beams
        
    @staticmethod
    def DBeam(database, beam_width, beam_length, dynamic_threshold, type = 'AC_EMBEDDING_SIMILARITY', table_scoring_type = 'SIMPLE', initializer = 'COSINE_SIMILARITY', verbose = False):
        if beam_width and dynamic_threshold:
            raise Exception("Cannot define both beam width and dynamic threshold")
        
        if dynamic_threshold and type != 'AC_EMBEDDING_SIMILARITY_V4':
            raise Exception("Dynamic threshold only works with AC_EMBEDDING_SIMILARITY_V4 algorithm")
        
        if beam_width and type == 'AC_EMBEDDING_SIMILARITY_V4':
            raise Exception("Beam width does not work with AC_EMBEDDING_SIMILARITY_V4 algorithm")
        
        Utils.calculate_table_scores(database, table_scoring_type)
        
        init_beams = Utils.initialize_beams(beam_width, database, dynamic_threshold, initializer = initializer)
            
        final_beams = Utils.calculate_beams(init_beams, beam_width, beam_length, dynamic_threshold, database, type, verbose = verbose)
        
        return set(final_beams[0].tables)
    
    @staticmethod
    def topN(database, n):
        tables_sorted = sorted([table for schema in database.schemata for table in schema.tables], key=lambda table: table.cosine_similarity, reverse=True)

        tables_selected = tables_sorted[:n]

        return set(tables_selected)
    
    @staticmethod
    def testModel(model, database, data, beam_width, beam_length, dynamic_threshold, type = 'AC_EMBEDDING_SIMILARITY', table_scoring_type = 'SIMPLE', initializer = 'COSINE_SIMILARITY', embedding_calculation = 'TABLE_DEFINITION'):
        Utils.calculate_table_embeddings(database, model, type = embedding_calculation)

        gold_schema_tables_sets = []
        dbeam_tables_selected_sets = []
        topn_tables_selected_sets =[]

        for i in range(0, len(data)):

            question = data['QUESTION'][i]
            Utils.calculate_tables_question_cosine_similarity(database, question, model)
            #Extract gold schema from data
            gold_schema_tables = set([database.find_table_by_name(tableName) for tableName in Utils.extract_tables_from_data(data['SCHEMA'][i])])

            dbeam_tables_selected = Utils.DBeam(database, beam_width, beam_length, dynamic_threshold, type, table_scoring_type, initializer = initializer, verbose = False)
            topn_tables_selected = Utils.topN(database, beam_length)

            gold_schema_tables_sets.append(gold_schema_tables)
            dbeam_tables_selected_sets.append(dbeam_tables_selected)
            topn_tables_selected_sets.append(topn_tables_selected)

        return Utils.extractDBeamMetrics(gold_schema_tables_sets, dbeam_tables_selected_sets, topn_tables_selected_sets)
    
    @staticmethod
    def extractDBeamMetrics(gold_schema_tables_sets, dbeam_tables_selected_sets, topn_tables_selected_sets):
        dbeam_accuracy1 = Utils.accuracy1(gold_schema_tables_sets, dbeam_tables_selected_sets)
        dbeam_accuracy2 = Utils.accuracy2(gold_schema_tables_sets, dbeam_tables_selected_sets)

        topn_accuracy1 = Utils.accuracy1(gold_schema_tables_sets, topn_tables_selected_sets)
        topn_accuracy2 = Utils.accuracy2(gold_schema_tables_sets, topn_tables_selected_sets)

        return [
            ['DBeam', dbeam_accuracy1[0], dbeam_accuracy1[1], dbeam_accuracy1[2], dbeam_accuracy2[2]],
            ['TopN', topn_accuracy1[0], topn_accuracy1[1], topn_accuracy1[2], topn_accuracy2[2]],
            Utils.perQuestionMetrics(gold_schema_tables_sets, dbeam_tables_selected_sets, topn_tables_selected_sets)
        ]
    
    @staticmethod
    def accuracy1(gold_schema_tables_sets, tables_selected_sets):
        total_correct = 0
        total_missed = 0

        for i in range(len(gold_schema_tables_sets)):
            gold_schema_tables = gold_schema_tables_sets[i]
            tables_selected = tables_selected_sets[i]
            if gold_schema_tables.issubset(tables_selected):
                total_correct += 1
            else:
                total_missed += 1

        return [total_correct, total_missed, round((total_correct / (total_correct + total_missed))*100, 3)]
    
    @staticmethod
    def accuracy2(gold_schema_tables_sets, tables_selected_sets):
        total_correct = 0
        total_missed = 0

        for i in range(len(gold_schema_tables_sets)):
            gold_schema_tables = gold_schema_tables_sets[i]
            tables_selected = tables_selected_sets[i]

            for gold_table in gold_schema_tables:
                if gold_table in tables_selected:
                    total_correct += 1
                else:
                    total_missed += 1

        return [total_correct, total_missed, round((total_correct / (total_correct + total_missed))*100, 3)]
    
    @staticmethod
    def perQuestionMetrics(gold_schema_tables_sets, dbeam_tables_selected_sets, topn_tables_selected_sets):
        
        dbeam = []
        topn = []
        output = []

        for i in range(len(gold_schema_tables_sets)):
            total_correct = 0
            gold_schema_tables = gold_schema_tables_sets[i]
            tables_selected = dbeam_tables_selected_sets[i]

            for gold_table in gold_schema_tables:
                if gold_table in tables_selected:
                    total_correct += 1
            
            dbeam.append(total_correct)

        for i in range(len(gold_schema_tables_sets)):
            total_correct = 0
            gold_schema_tables = gold_schema_tables_sets[i]
            tables_selected = topn_tables_selected_sets[i]

            for gold_table in gold_schema_tables:
                if gold_table in tables_selected:
                    total_correct += 1
            
            topn.append(total_correct)
            
        for i in range(len(gold_schema_tables_sets)):
            total_question_tables = len(gold_schema_tables_sets[i])
            total_dbeam = dbeam[i]
            total_topn = topn[i]
            output.append([total_question_tables, total_dbeam, total_topn])
        
        return output
        
        
            
    
    @staticmethod
    def blockPrint():
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return original_stdout

    @staticmethod
    def enablePrint(std_out):
        sys.stdout = std_out

class Database:
    def __init__(self, name, schemata = []):
        self.name = name
        self.schemata = schemata
    
    def get_definition(self, ignore_schema = False, dependency_valid = False):
        definition = ''
        if not ignore_schema:
            for schema in self.schemata:
                definition += "CREATE SCHEMA " + schema.name + ';\n'

        definition += '\n'

        if not dependency_valid:

            for schema in self.schemata:
              for table in schema.tables:
                definition += table.get_definition(ignore_descriptions = True, single_line = False, ignore_table_constraints = False, ignore_field_constraints = False, ignore_primary_key_constraints = False, ignore_foreign_key_constraints = False, ignore_schema = ignore_schema) + '\n\n'
        
        if dependency_valid:
            tables = []
            
            #First all tables that don't have relations
            for schema in self.schemata:
                for table in schema.tables:
                    if not table.relations:
                        tables.append(table)

            #While not all tables added
            while set(tables) != set([table for schema in self.schemata for table in schema.tables]):
                #Iterate through all the tables and add all tables that are not already added and have their dependencies satisfied
                for schema in self.schemata:
                    for table in schema.tables:
                        if table not in tables:

                            dependencyValid = True
                            for relation in table.relations:
                                if relation.toTableName not in [table.name for table in tables]:
                                   dependencyValid = False

                            if dependencyValid:
                                tables.append(table)
                                
            for table in tables:
                definition += table.get_definition(ignore_descriptions = True, single_line = False, ignore_table_constraints = False, ignore_field_constraints = False, ignore_primary_key_constraints = False, ignore_foreign_key_constraints = False, ignore_schema = ignore_schema) + '\n\n'

        return definition
    
    def get_dependecy_valid_table_order(self):
        tables = []
            
        #First all tables that don't have relations
        for schema in self.schemata:
            for table in schema.tables:
                if not table.relations:
                    tables.append(table)
        #While not all tables added
        while set(tables) != set([table for schema in self.schemata for table in schema.tables]):
            #Iterate through all the tables and add all tables that are not already added and have their dependencies satisfied
            for schema in self.schemata:
                for table in schema.tables:
                    if table not in tables:
                        dependencyValid = True
                        for relation in table.relations:
                            if relation.toTableName not in [table.name for table in tables]:
                               dependencyValid = False
                        if dependencyValid:
                            tables.append(table)
        
        return tables
    
    def find_table_by_name(self, tableName):
        for schema in self.schemata:
            for table in schema.tables:
                if table.name == tableName:
                    return table
        raise Exception("Table with name: " + tableName + " not found")
    
    def get_tables_referencing_table(self, table):
        refering_tables = []
        for schema in self.schemata:
            for schemaTable in schema.tables:
                if schemaTable.name == table.name:
                   continue
                if schemaTable.relations:
                   for relation in schemaTable.relations:
                      if relation.toTableName == table.name:
                         refering_tables.append(schemaTable)

        return refering_tables
    
    def get_referenced_tables(self, table):
        referenced_tables = []
        if table.relations:
            for relation in table.relations:
               referenced_tables.append(self.find_table_by_name(relation.toTableName))

        return referenced_tables
       


class Schema:
    def __init__(self, name, tables = []):
        self.name = name
        self.tables = tables

class TableField:
    def __init__(self, name, type, constraints = None, description = None):
        self.name = name
        self.type = type
        self.constraints = constraints
        self.description = description
    
    def get_definition(self, lineEnd = ' ', ignore_descriptions = False, ignore_field_constraints = False, ignore_primary_key_constraints = False):
        constraints = ''
        if self.constraints:
            for constraint in self.constraints:
                if ignore_field_constraints and not constraint.startswith('PRIMARY KEY'): 
                    continue

                if ignore_primary_key_constraints and constraint.startswith('PRIMARY KEY'):
                   continue

                constraints += ' ' + constraint

        lineComment = ''
        if self.description and not ignore_descriptions:
            lineComment = '-- ' + self.description + ' '
        
        return self.name + ' ' + self.type + constraints + lineEnd + lineComment
       

class TableRelation:
    def __init__(self, toTable, fromField, toField):
        self.toTable = toTable
        self.fromField = fromField
        self.toField = toField
    
    def __init__(self, toTableSchemaName, toTableName, fromFieldName, toFieldName, constraint):
        self.toTableSchemaName = toTableSchemaName
        self.toTableName = toTableName
        self.fromFieldName = fromFieldName
        self.toFieldName = toFieldName
        self.constraint = constraint
    
    def get_definition(self, ignore_schema = False):
        toTableDefinition = self.toTableSchemaName + '.' + self.toTableName
        fromFieldsDefinition = ''
        toFieldsDefinition = ''

        if isinstance(self.fromFieldName, str):
           fromFieldsDefinition = '(' + self.fromFieldName + ')'
        
        if isinstance(self.fromFieldName, list):
            fromFieldsDefinition += '('
            for i in range(0, len(self.fromFieldName)):
                fromFieldsDefinition += self.fromFieldName[i]
                if i != len(self.fromFieldName) - 1:
                    fromFieldsDefinition += ', '
            fromFieldsDefinition += ')'
        
        if isinstance(self.toFieldName, str):
           toFieldsDefinition = '(' + self.toFieldName + ')'
        
        if isinstance(self.toFieldName, list):
            toFieldsDefinition += '('
            for i in range(0, len(self.toFieldName)):
                toFieldsDefinition += self.toFieldName[i]
                if i != len(self.toFieldName) - 1:
                    toFieldsDefinition += ', '
            toFieldsDefinition += ')'
        
        if ignore_schema:
            toTableDefinition = self.toTableName
        return 'FOREIGN KEY ' + fromFieldsDefinition + ' REFERENCES ' + toTableDefinition + toFieldsDefinition

class Table:
    def __init__(self, schema, name, definition):
        self.schema = schema
        self.name = name
        self.definition = definition
    
    def __init__(self, schemaName, name, fields, relations = None, constraints = None, description = None):
        self.schemaName = schemaName
        self.name = name
        self.fields = fields
        self.relations = relations
        self.constraints = constraints
        self.description = description

    @staticmethod
    def get_tables_from_parsed_schema_map(schemas_tables_tables_definition_map):
        tables = []
        for schema, tables in schemas_tables_tables_definition_map.items():
            for table, definition in tables.items():
                tables.append(Table(schema, table, definition))
        return tables

    @staticmethod
    def get_tables_from_schema(schema_path):
        schemas_tables_tables_definition_map = Utils.get_schema_table_table_definition_map(schema_path)
        table_objects = []
        for schema, tables in schemas_tables_tables_definition_map.items():
            for table, definition in tables.items():
                table_objects.append(Table(schema, table, definition))
        return table_objects
    
    def get_definition(self, ignore_descriptions = False, single_line = False, ignore_table_constraints = False, ignore_field_constraints = False, ignore_primary_key_constraints = False, ignore_foreign_key_constraints = False, ignore_schema = False, fields_to_ignore = []):
        definition = ''
        tableComment = ''

        #Cannot have comments in single line
        if single_line:
           ignore_descriptions = True

        if self.description and not ignore_descriptions:
          tableComment = '-- ' + self.description + ' '
        
        newline = '\n'
        if single_line:
           newline = ''
        
        if ignore_schema:
            definition += "CREATE TABLE " + self.name + '( ' + tableComment + newline
        else:
            definition += "CREATE TABLE " + self.schemaName + '.' + self.name + '( ' + tableComment + newline

        #Filter fields
        fieldsFiltered = []
        for field in self.fields:
           if field.name not in fields_to_ignore:
              fieldsFiltered.append(field)

        #Filter constraints
        #Foreign keys will be added from self.relations fields
        constraintsFiltered = None
        if self.constraints:
            for constraint in self.constraints:
                if not (constraint.startswith('CONSTRAINT') and ignore_table_constraints) and not (constraint.startswith('FOREIGN KEY')) and not (constraint.startswith('PRIMARY KEY') and ignore_primary_key_constraints):
                    if not constraintsFiltered:
                       constraintsFiltered = []
                    constraintsFiltered.append(constraint)
        
        #Filter relations
        relationsFiltered = None
        if self.relations:  
            for relation in self.relations:
               if relation.fromFieldName not in fields_to_ignore:
                    if not relationsFiltered:
                        relationsFiltered =[]
                    relationsFiltered.append(relation)

        #Create fields
        for i in range(0, len(fieldsFiltered)):
            field = fieldsFiltered[i]

            lineEnd = ', '
            if i == len(fieldsFiltered) - 1 and not constraintsFiltered and not relationsFiltered:
                lineEnd = ' '
            
            newline = '\n'
            if single_line:
               newline = ''
            
            startLine = '\t'
            if single_line:
               startLine = ''
            
            definition += startLine + field.get_definition(lineEnd, ignore_descriptions, ignore_field_constraints, ignore_primary_key_constraints) + newline

        #Create constraints
        if constraintsFiltered:
          for i in range(0, len(constraintsFiltered)):
            constraint = constraintsFiltered[i]
            
            lineEnd = ', '
            if i == len(constraintsFiltered) - 1 and not relationsFiltered:
              lineEnd = ' '
            
            newline = '\n'
            if single_line:
               newline = ''

            startLine = '\t'
            if single_line:
               startLine = ''

            definition += startLine + constraint + lineEnd + newline

        #Add foreign keys
        if relationsFiltered:
           for i in range(0, len(relationsFiltered)):
                relation = relationsFiltered[i].get_definition(ignore_schema = ignore_schema)

                lineEnd = ', '
                if i == len(relationsFiltered) - 1:
                  lineEnd = ' '

                newline = '\n'
                if single_line:
                   newline = ''
                
                startLine = '\t'
                if single_line:
                   startLine = ''
                
                definition += startLine + relation + lineEnd + newline

        definition += '); '

        return definition
    
    def get_description_including_columns(self):
        description = self.description
        for field in self.fields:
            if field.description:
                description += ' ' + field.description
        return description 
  
    def get_definition_cls_embedding(self, model, tokenizer):
        print(self.name)
        return Utils.get_embeddings(self.definition, model, tokenizer, 512)[0]
    
    def get_scores_for_dbeam(self):
        score_print = self.name + '('
        if hasattr(self, 'cosine_similarity') and self.cosine_similarity:
            score_print += str(round(self.cosine_similarity, 3)) + ', '
        if hasattr(self, 'neighbor_score') and self.neighbor_score:
            score_print += str(round(self.neighbor_score, 3)) + ', '
        if hasattr(self, 'score') and self.score:
            score_print += str(round(self.score, 3)) + ', '
        score_print += ')'
        return score_print
    
class Beam:
    def __init__(self):
        self.tables = []

    def to_string(self):
        for table in self.tables:
            print(table.name)

