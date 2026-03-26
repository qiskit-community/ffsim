{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods %}
   .. automethod:: __init__

   {%- set is_protocol = '__protocol_attrs__' in members and '__protocol_attrs__' not in inherited_members %}
   {%- set ns = namespace(protocol_methods=[]) %}
   {%- if is_protocol %}
   {%- for item in all_methods %}
   {%- if item.startswith('_') and not item.startswith('__') and item.endswith('_') and not item.endswith('__') %}
   {%- set ns.protocol_methods = ns.protocol_methods + [item] %}
   {%- endif %}
   {%- endfor %}
   {%- endif %}
   {% if methods or ns.protocol_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {%- for item in ns.protocol_methods if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {%- for item in ns.protocol_methods %}

   .. automethod:: {{ item }}
   {%- endfor %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
