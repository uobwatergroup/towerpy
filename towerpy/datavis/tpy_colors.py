"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

"""
This file contains a dictionary listing different colour gradients.
"""

towerpy_colours = {
    'rad_ref': ["#7665A5", "#9483BE", "#B2A5D4", "#D2C9E8",  # lavender block
                "#7ACEEA", "#63BDE4", "#4EACDD", "#3999D3", "#2A82C4",
                "#1C6BB0",  # blues
                "#EED97A","#E7C355", "#DFAC3A", "#D49328", "#C87A1C",
                # yellows->orange
                "#D06A3A", "#C0543A", "#A6453F", "#8A3A45", "#542837",
                # warm->plum
                ],
    'rad_pvars': ["#C7E7F6", "#A9D9F0", "#8BCBE8", "#6CB9DD", "#4DA4CF",
                  "#2F8EC0",  # blues
                  "#EED97A", "#E7C355", "#DFAC3A", "#D49328", "#C87A1C",
                  # yellows->orange
                  "#C45A3A", "#A24A57", "#7F3F6A", "#6D3A64", "#5C345F",
                  "#422A4C",  # warm->plum
                  "#28182F",  # deep plum
                  # "#857C97",  # lavender‑grey
                  "#746783",  # lavender‑grey_b
                  "#9F97B2",  # final lavender‑grey tail
                  ],
    'rad_rainrt': ["#A9D9F0", "#8BCBE8", "#6CB9DD", "#4DA4CF", "#2F8EC0",
                   # blues
                   "#6FBDBE", "#4EAAA9", "#2E9594", "#1F7F7F", "#166A6A",
                   # teals
                   "#EED97A", "#E7C355", "#DFAC3A", "#D49328", "#C87A1C",
                   # yellows->orange
                   "#C45A3A", "#A24A57", "#7F3F6A", "#5C345F", "#422A4C",
                   # warm->plum
                   "#9F97B2"  # lavender tail
                   ],
    'rad_2slope': ["#3F4446", "#586369", "#74878E", "#93ADB6", "#9DBBC4",
                    # greys
                   "#A8C6D0", "#7ACEEA","#63BDE4", "#4EACDD", "#3999D3",
                   "#2A82C4",  # blues
                   "#EED97A", "#E7C355", "#DFAC3A", "#D49328", "#C87A1C",
                   # yellows->orange
                   "#C45A3A", "#A24A57", "#7F3F6A", "#5C345F", "#422A4C",
                   # warm->plum
                   "#9F97B2"  # lavender tail
                   ],
    'rad_model': ['#ffffff', '#efefef', '#dfdfdf', '#cfcfcf', '#bfbfbf',
                  '#b0b0b0', '#a1a1a1', '#929292', '#838383', '#757575',

                  '#00305a', '#113f6b', '#204f7d', '#2d608e', '#3a71a0',
                  '#4783b3', '#5595c5', '#62a8d8', '#70bbea', '#7ecefd',

                  '#ffee58', '#ffe148', '#ffd339', '#ffc52a', '#ffb71b',
                  '#ffa80c', '#ff9900', '#ff8900', '#ff7800', '#ff6600',

                  '#ff1d23', '#ec1820', '#d9141d', '#c60f1a', '#b30b17',
                  '#a10714', '#8f0411', '#7e020d', '#6d0108', '#5c0002',

                  '#3a0f57', '#4a1e62', '#592c6d', '#683a78', '#784984',
                  '#87588f', '#96689b', '#a677a8', '#b587b4', '#c498c1'],
    'div_yw_gy_bu': ['#509AE8', '#505050', '#FFBA01'],
    'div_dbu_rd': ['#DA2C43', '#E15566', '#E97E88', '#F0A8AB', '#F8D1CD',
                   '#AFC6D9', '#83A3BE', '#5880A2', '#2C5D87', '#003A6B'],
    'div_lbu_w_rd': ['#DA2C43', '#E15566', '#E97E88', '#F0A8AB', '#F8D1CD',
                     '#FFFAF0', '#F0FFFE', '#C0E5FE', '#90CCFE', '#60B2FE',
                     '#3099FE', '#007FFE'],
    'div_dbu_w_rd': ['#DA2C43', '#E15566', '#E97E88', '#F0A8AB', '#F8D1CD',
                     '#FFFAF0', '#F0FFFE', '#AFC6D9', '#83A3BE', '#5880A2',
                     '#2C5D87', '#003A6B'],
    'div_dbu_w_k': ['#2B2B2B', '#545454', '#808080', '#AFAFAF', '#E1E1E1',
                    '#FCFFFA', '#F0FFFE', '#AFC6D9', '#83A3BE', '#5880A2',
                    '#2C5D87', '#003A6B'],
    'div_rd_w_k': ['#2B2B2B', '#545454', '#808080', '#AFAFAF', '#E1E1E1',
                   '#FCFFFA', '#FFFAF0', '#F8D1CD', '#F0A8AB', '#E97E88',
                   '#E15566', '#DA2C43'],
    'useq_grey': ['#505050', '#5b5d5e', '#666a6c', '#71777b', '#7b8489',
                  '#869197', '#919ea5', '#9cabb3', '#a7b8c1', '#b1c5d0',
                  '#bcd2de', '#c7dfec', '#d2ecfa'],
    'useq_ywbu': ['#FFBA01', '#d0a819', '#99872B', '#666D41', '#335456',
                  '#003A6B'],
    'useq_bupkyw': ['#3F3A81', '#954698', '#D65A94', '#F76385', '#F9B357',
                    '#F8F658'],
    'useq_morning': ['#F6BD73', '#F9D69E', '#CBB6B0', '#9DA3B7', '#5373A1',
                     '#3D5688'],
    'useq_wblk': ['#F2F3F4', '#B3BDD4', '#7587B4', '#365194', '#253354',
                  '#141414'],
    'useq_fiery': ['#FCD988', '#F9A622', '#F55E01', '#EB2701', '#C40000',
                   '#292929'],
    'useq_tec': ['#B0F7FF', '#9BCFE3', '#86A7C7', '#7080AA', '#5B588E',
                 '#463072'],
    'useq_pastel': ['#FCF5E3', '#FCD4B8', '#F3AE9C', '#D799A7', '#9A7DA7',
                    '#4E60A4'],
    'useq_wk': ['#F8F8FF', '#CAC9CD', '#9B9A9C', '#6D6A6A', '#3E3B39',
                '#100C07'],
    'useq_sun': ['#FFF673', '#FFDB02', '#FFBC11', '#FF9D1F', '#FF7E2E',
                 '#FF6352'],
    'useq_calm': ['#F6EEAB', '#C9DD94', '#9DCF94', '#7EC796', '#5EBD96',
                  '#11A797'],
    }
