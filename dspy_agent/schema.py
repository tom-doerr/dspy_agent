"""XML schema definitions for the unified agent module."""

# Input schema - what the agent receives
INPUT_XML_SCHEMA = """
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:element name="agent_state">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="memory" type="xs:string" />
        <xs:element name="last_plan" type="xs:string" />
        <xs:element name="last_action" type="xs:string" />
        <xs:element name="observation" type="xs:string" />
      </xs:sequence>
    </xs:complexType>
  </xs:element>
</xs:schema>
"""

# Plan schema - defines the structure of a plan
PLAN_XML_SCHEMA = """
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:element name="plan">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="goal" type="xs:string" />
        <xs:element name="steps">
          <xs:complexType>
            <xs:sequence>
              <xs:element name="step" maxOccurs="unbounded">
                <xs:complexType>
                  <xs:sequence>
                    <xs:element name="action" type="xs:string" />
                    <xs:element name="reason" type="xs:string" minOccurs="0" />
                  </xs:sequence>
                  <xs:attribute name="id" type="xs:integer" use="required" />
                </xs:complexType>
              </xs:element>
            </xs:sequence>
          </xs:complexType>
        </xs:element>
        <xs:element name="current_step_id" type="xs:integer" />
      </xs:sequence>
    </xs:complexType>
  </xs:element>
</xs:schema>
"""

# Execution instructions schema - defines the structure of execution instructions
EXECUTION_XML_SCHEMA = """
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:element name="write_operations">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="operation" maxOccurs="unbounded">
          <xs:complexType>
            <xs:simpleContent>
              <xs:extension base="xs:string">
                <xs:attribute name="type" use="required">
                  <xs:simpleType>
                    <xs:restriction base="xs:string">
                      <xs:enumeration value="command" />
                      <xs:enumeration value="file" />
                      <xs:enumeration value="message" />
                    </xs:restriction>
                  </xs:simpleType>
                </xs:attribute>
                <xs:attribute name="path" type="xs:string" />
                <xs:attribute name="command" type="xs:string" />
              </xs:extension>
            </xs:simpleContent>
          </xs:complexType>
        </xs:element>
      </xs:sequence>
    </xs:complexType>
  </xs:element>
</xs:schema>
"""

# Output schema - what the agent produces
OUTPUT_XML_SCHEMA = """
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:element name="agent_output">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="updated_memory" type="xs:string" />
        <xs:element name="new_plan">
          <xs:complexType>
            <xs:sequence>
              <xs:any processContents="skip" />
            </xs:sequence>
          </xs:complexType>
        </xs:element>
        <xs:element name="execution_instructions">
          <xs:complexType>
            <xs:sequence>
              <xs:any processContents="skip" />
            </xs:sequence>
          </xs:complexType>
        </xs:element>
        <xs:element name="expected_outcome" type="xs:string" />
        <xs:element name="is_done" type="xs:boolean" />
      </xs:sequence>
    </xs:complexType>
  </xs:element>
</xs:schema>
"""

# Example input XML
EXAMPLE_INPUT_XML = """
<agent_state>
  <memory>Previous knowledge about the task</memory>
  <last_plan><action>previous_action</action></last_plan>
  <last_action>command that was executed</last_action>
  <observation>Result of the last action or new information</observation>
</agent_state>
"""

# Example plan XML
EXAMPLE_PLAN_XML = """
<plan>
  <goal>Find all Python files in the current directory</goal>
  <steps>
    <step id="1">
      <action>List all files in the current directory</action>
      <reason>To identify all files before filtering for Python files</reason>
    </step>
    <step id="2">
      <action>Filter the list to only include Python files</action>
      <reason>To focus only on files with .py extension</reason>
    </step>
    <step id="3">
      <action>Count the number of Python files</action>
      <reason>To provide a summary of findings</reason>
    </step>
  </steps>
  <current_step_id>1</current_step_id>
</plan>
"""

# Example execution instructions XML
EXAMPLE_EXECUTION_XML = """
<write_operations>
  <operation type="command" command="ls -la">ls -la</operation>
  <operation type="message">Listing all files in the current directory</operation>
  <operation type="file" path="results.txt">Contents to write to the file</operation>
</write_operations>
"""

# Example output XML
EXAMPLE_OUTPUT_XML = """
<agent_output>
  <updated_memory>Previous knowledge about the task. Found 5 Python files in the current directory.</updated_memory>
  <new_plan>
    <plan>
      <goal>Find all Python files in the current directory</goal>
      <steps>
        <step id="1">
          <action>List all files in the current directory</action>
          <reason>To identify all files before filtering for Python files</reason>
        </step>
        <step id="2">
          <action>Filter the list to only include Python files</action>
          <reason>To focus only on files with .py extension</reason>
        </step>
        <step id="3">
          <action>Count the number of Python files</action>
          <reason>To provide a summary of findings</reason>
        </step>
      </steps>
      <current_step_id>2</current_step_id>
    </plan>
  </new_plan>
  <execution_instructions>
    <write_operations>
      <operation type="command" command="find . -name '*.py'">find . -name '*.py'</operation>
      <operation type="message">Filtering for Python files only</operation>
    </write_operations>
  </execution_instructions>
  <expected_outcome>Discovery of all Python source files in current directory with accurate count</expected_outcome>
  <is_done>false</is_done>
</agent_output>
"""
