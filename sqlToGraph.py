import re
import networkx as nx

def add_tables_to_graph(schema_tables, graph):
    for schema, tables in schema_tables.items():
        for table in tables.keys():
            node_name = f"{schema}.{table}"
            graph.add_node(node_name)

def add_edges_to_graph(schema_tables, graph):
    # Adjusting the regular expression to match foreign key constraints within single-line definitions
    # This pattern looks for the "REFERENCES" keyword followed by the schema and table name (considering schema names in references)
    foreign_key_pattern = re.compile(r"REFERENCES (\w+)\.(\w+)")

    for schema, tables in schema_tables.items():
        for table, definition in tables.items():
            # Find all matches for foreign key constraints within the table definition
            foreign_keys = foreign_key_pattern.findall(definition)

            for ref_schema, ref_table in foreign_keys:
                # Since the schema name is now important, we include it in the node names for both the current table and the referenced table
                # This way, we can accurately represent relationships between tables across different schemas
                graph.add_edge(f"{schema}.{table}", f"{ref_schema}.{ref_table}")