from neo4j import GraphDatabase
import os
import json


class Database:
    """ Singleton class for accessing a Neo4j database. """

    database_config_path = 'utils/helpers/database_config.json'
    _instance: 'Database' = None


    def __new__(cls) -> 'Database':
        """ Establish a database connection. """

        if cls._instance is None:
            cls._instance = super().__new__(cls)

            save_config = False

            if not os.path.exists(cls.database_config_path):
                print('[!] Neo4j database config not found.')
                cls._instance.uri = input(' |  > Enter URI [bolt://localhost:7687]: ') or 'bolt://localhost:7687'
                cls._instance.username = input(' |  > Enter Username [neo4j]: ') or 'neo4j'
                cls._instance.password = input(' |  > Enter Password [password]: ') or 'password'
                cls._instance.database = input(' |  > Enter DB Name [neo4j]: ') or 'neo4j'
                save_config = True

            else:
                print('[i] Neo4j database config found.')
                with open(cls.database_config_path, 'r') as file:
                    config = json.load(file)
                    cls._instance.uri = config['uri']
                    cls._instance.username = config['username']
                    cls._instance.password = config['password']
                    cls._instance.database = config['database']

            print(f'[i] Attempting connection...')
            cls._instance.driver = GraphDatabase.driver(
                uri = cls._instance.uri, auth = (cls._instance.username, cls._instance.password)
            )
            cls._instance.driver.verify_connectivity()

            if save_config:
                with open(cls.database_config_path, 'w') as file:
                    json.dump({
                        'uri' : cls._instance.uri,
                        'username' : cls._instance.username,
                        'password' : cls._instance.password,
                        'database' : cls._instance.database
                    }, file)

            print('[+] Database connection established.')

        return cls._instance


    def query(self, query: str, params: dict[str, ...] = None) -> dict[str, ...] | list[dict[str, ...]]:
        """
        Execute a database query.

        Arguments:
            query: The query.
            params: Query parameters.

        Returns:
            The result of the query.
        """

        if params:
            result = self.driver.execute_query(query, params, database_ = self.database)
        else:
            result = self.driver.execute_query(query, database_ = self.database)

        if len(result.records) == 1:
            return result.records[0].data()

        return [record.data() for record in result.records]


__all__ = ['Database']