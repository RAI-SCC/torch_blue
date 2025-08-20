{% if obj.display %}
   {% if is_own_page %}
{{ obj.id }}
{{ "=" * obj.id | length }}

   {% endif %}
.. py:method:: {% if is_own_page %}{{ obj.id }}{% else %}{{ obj.short_name }}{% endif %}({{ obj.args|custom_filter_type_hints }}){% if obj.return_annotation is not none %} -> {{ obj.return_annotation|custom_filter_type_hints }}{% endif %}
   {% for (args, return_annotation) in obj.overloads %}

               {%+ if is_own_page %}{{ obj.id }}{% else %}{{ obj.short_name }}{% endif %}({{ args }}){% if return_annotation is not none %} -> {{ return_annotation }}{% endif %}
   {% endfor %}
   {% for property in obj.properties %}

   :{{ property }}:
   {% endfor %}

   {% if obj.docstring %}

   {{ obj.docstring|indent(3) }}
   {% endif %}
{% endif %}
