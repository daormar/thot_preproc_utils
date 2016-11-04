"""
This file collects functional tests concerning behaviours related to
externally modifying terms. External force might be QA Team via terms API.

In the future the scope might be broadened. Modify this documentation then.
The scope is narrow at the time of writing this as this is the first
functional test package.
"""

from thot_utils.libs.thot_preproc import annotated_string_to_xml_skeleton


def test_string_with_tags():
    s = '<length_limit>80</length_limit>320GB SATA 2.5" ' \
        'Hard Disc Drive Upgrade For HP Compaq Presario CQ42-108TU Laptop'
    expected = [
        [True, '<length_limit>'],
        [False, '80'],
        [True, '</length_limit>'],
        [False, '320GB SATA 2.5" Hard Disc Drive Upgrade For HP Compaq Presario CQ42-108TU Laptop']
    ]
    assert list(annotated_string_to_xml_skeleton(s)) == expected
